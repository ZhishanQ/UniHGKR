import json
import re
import time
import requests

ENT_PATTERN = re.compile("^Q[0-9]+$")
PRE_PATTERN = re.compile("^P[0-9]+$")

NUMBER_PATTERN = re.compile("^[0-9]+$")

YEAR_PATTERN = re.compile("^[0-9][0-9][0-9][0-9]$")
DATE_PATTERN = re.compile("^[0-9]+ [A-z]+ [0-9][0-9][0-9][0-9]$")
DATE_PATTERN_MDY = re.compile("[A-Z][a-z]* [0-9]+, [0-9][0-9][0-9][0-9]")

TIMESTAMP_PATTERN_1 = re.compile('^"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z"')
TIMESTAMP_PATTERN_2 = re.compile("^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z")


class StringLibrary:
    def __init__(self, config):
        # load stopwords
        with open(config["path_to_stopwords"], "r") as fp:
            self.stopwords = fp.read().split("\n")

        # create session for faster connections
        self.request_session = requests.Session()

    def get_question_words(self, question, ner=None, nlp=None):
        """
        Extracts a list of question words from the question.
        Named entity phrases can be detected by the specified ner method.
        If 'ner' is set to False, each word is considered individually.
        Stopwords, symbols and punctuations are removed.
        """
        question_words = []
        # apply NER
        entity_spots = self._apply_NER(question, ner, nlp)
        for spot in entity_spots:
            if spot.lower() in self.stopwords:
                continue
            question = question.replace(spot, "")
            question_words.append(spot)
        # remove symbols
        question = (
            question.replace(",", "")
            .replace("!", "")
            .replace("?", "")
            .replace(".", "")
            .replace("'", "")
            .replace('"', "")
            .replace(":", "")
            .replace("’", "")
            .replace("{", "")
            .replace("}", "")
        )
        # expand the question by whitespaces to be able to find the stopwords
        question = (" " + question + " ").lower()
        # remove stopwords
        for stopword in self.stopwords:
            while " " + stopword + " " in question:
                question = question.replace(" " + stopword + " ", " ")
        # remove remaining s from plural or possesive expressions
        question = question.replace(" s ", " ")
        # remove double whitespaces
        while "  " in question:
            question = question.replace("  ", " ")
        # remove the whitespace(s) at the front and end
        question = question.strip()
        # get all question words
        question_words += question.split(" ")
        question_words = [
            question_word for question_word in question_words if question_word.strip()
        ]
        return question_words

    def _apply_NER(self, question, ner="tagme", nlp=None):
        """
        Apply the given NER method on the question.
        Returns all detected entity mentions.
        """
        if ner is None:
            return []

    @staticmethod
    def convert_month_to_number(month):
        """Map the given month to a number."""
        return {
            "january": "01",
            "february": "02",
            "march": "03",
            "april": "04",
            "may": "05",
            "june": "06",
            "july": "07",
            "august": "08",
            "september": "09",
            "october": "10",
            "november": "11",
            "december": "12",
        }[month.lower()]

    @staticmethod
    def convert_number_to_month(number):
        """Map the given month to a number."""
        return {
            "01": "January",
            "02": "February",
            "03": "March",
            "04": "April",
            "05": "May",
            "06": "June",
            "07": "July",
            "08": "August",
            "09": "September",
            "10": "October",
            "11": "November",
            "12": "December",
        }[number]

    @staticmethod
    def wikidata_url_to_wikidata_id(url):
        """Extract the wikidata id from a wikidata url."""
        if not url:
            return False
        # xml date
        if "XMLSchema#dateTime" in url or "XMLSchema#decimal" in url:
            date = url.split('"', 2)[1]
            date = date.replace("+", "")
            return date
        # literal
        if not ("wikidata.org" in url):
            # year
            if StringLibrary.is_year(url):
                return StringLibrary.convert_year_to_timestamp(url)
            # dmy date
            elif StringLibrary.is_date(url):
                return StringLibrary.convert_date_to_timestamp(url)
            # mdy date
            elif StringLibrary.is_mdy_date(url):
                return StringLibrary._convert_mdy_to_timestamp(url)
            # is yes (existential question)
            elif StringLibrary.is_existential_yes(url):
                return "Yes"
            # is no (existential question)
            elif StringLibrary.is_existential_no(url):
                return "No"
            # other constant
            else:
                url = url.replace('"', "")  # remove quotes
                return url
        # actual wikidata url
        else:
            # the wikidata id is always in the last component of the id
            wikidata_id = url.split("/")[-1]
            return wikidata_id

    @staticmethod
    def convert_year_to_timestamp(year):
        """Convert a year to a timestamp style."""
        return f"{year}-01-01T00:00:00Z"

    @staticmethod
    def convert_date_to_timestamp(date, date_format="dmy"):
        """Convert a date from the Wikidata frontendstyle to timestamp style."""
        try:
            if date_format == "dmy":
                return StringLibrary._convert_dmy_to_timestamp(date)
            else:
                return StringLibrary._convert_mdy_to_timestamp(date)
        except:
            print(
                f"The following date could not be parsed: {date}. Exception catched in string library (l.204)."
            )
            return date

    @staticmethod
    def _convert_dmy_to_timestamp(date):
        """
        Convert a date in dmy format to timestamp style.
        dmy format: https://en.wikipedia.org/wiki/Template:Use_dmy_dates
        """
        adate = date.split(" ")
        # add the leading zero
        if len(adate[0]) < 2:
            adate[0] = f"0{adate[0]}"
        # create timestamp
        year = adate[2]
        month = StringLibrary.convert_month_to_number(adate[1])
        day = adate[0]
        timestamp = f"{year}-{month}-{day}T00:00:00Z"
        return timestamp

    @staticmethod
    def _convert_mdy_to_timestamp(date):
        """
        Convert a date in mdy format to timestamp style.
        mdy format: https://en.wikipedia.org/wiki/Template:Use_mdy_dates
        """
        adate = date.split(" ")
        # remove comma and add the leading zero
        adate[1] = adate[1].replace(",", "")
        if len(adate[1]) < 2:
            adate[1] = f"0{adate[1]}"
        # create timestamp
        year = adate[2]
        month = StringLibrary.convert_month_to_number(adate[0])
        day = adate[1]
        timestamp = f"{year}-{month}-{day}T00:00:00Z"
        return timestamp

    @staticmethod
    def _convert_timestamp_to_date(timestamp):
        """Convert the given timestamp to the corresponding date."""
        adate = timestamp.split("-")
        # parse data
        year = adate[0]
        month = StringLibrary.convert_number_to_month(adate[1])
        day = adate[2].split("T")[0]
        # remove leading zero
        if day[0] == "0":
            day = day[1]
        if day == "1" and adate[1] == "01":
            # return year for 1st jan
            return year
        date = f"{day} {month} {year}"
        return date

    @staticmethod
    def is_entity(string):
        """Return if the given string is an entity id."""
        return ENT_PATTERN.match(string.strip()) != None

    @staticmethod
    def is_date(string):
        """Return if the given string is a date."""
        return DATE_PATTERN.match(string.strip()) != None

    @staticmethod
    def is_mdy_date(string):
        """
        Return if the given string is a mdy date.
        mdy format: https://en.wikipedia.org/wiki/Template:Use_mdy_dates
        """
        return DATE_PATTERN_MDY.match(string.strip()) != None

    @staticmethod
    def is_year(string):
        """Return if the given string describes a year in the format YYYY."""
        return YEAR_PATTERN.match(string.strip()) != None

    @staticmethod
    def is_number(string):
        """Return if the given string is a number."""
        string = string.replace('"', "").replace("+", "")
        string = string.strip()
        return NUMBER_PATTERN.match(string.strip()) != None

    @staticmethod
    def is_timestamp(string):
        """Return if the given string is a timestamp."""
        if TIMESTAMP_PATTERN_1.match(string.strip()) or TIMESTAMP_PATTERN_2.match(string.strip()):
            return True
        else:
            return False

    @staticmethod
    def is_existential_yes(string):
        """Return if the given string is a "Yes" answer to an existential question."""
        return string.strip().lower() == "yes"

    @staticmethod
    def is_existential_no(string):
        """Return if the given string is a "No" answer to an existential question."""
        return string.strip().lower() == "no"

    @staticmethod
    def get_year(timestamp):
        """Extract the year from the given timestamp."""
        return timestamp.split("-")[0]

    @staticmethod
    def parse_answers_to_ids(answer_urls):
        """
        Takes ConvQuestions answer_urls as input,
        returns list of parsed answer Wikidata ids.
        """
        return [
            StringLibrary.wikidata_url_to_wikidata_id(answer_url)
            for answer_url in answer_urls.split(";")
            if answer_url.strip()
        ]

    @staticmethod
    def parse_answers_to_dicts(answer_urls, labels_dict):
        """
        Takes ConvQuestions answer_urls as input,
        returns list of parsed answer dicts.
        """
        # keep original answers to keep information about years/dates,...
        original_answers = answer_urls.split(";")
        answers = StringLibrary.parse_answers_to_ids(answer_urls)

        answer_dicts = list()
        for i, id_ in enumerate(answers):
            if StringLibrary.is_entity(id_):
                answer = {"id": id_, "label": StringLibrary.item_to_label(id_, labels_dict)}
            else:
                # for non-entities, print original answer URL
                answer = {"id": id_, "label": original_answers[i]}
            answer_dicts.append(answer)

        return answer_dicts

    @staticmethod
    def item_to_label(item_id, labels_dict):
        """Retrieve one label for given item, without loading full CLOCQ-KB."""
        # if item_id is a timestamp, label is original date
        if TIMESTAMP_PATTERN_2.match(item_id):
            date = StringLibrary._convert_timestamp_to_date(item_id)
            return date

        # look-up item_id
        labels = labels_dict.get(item_id)
        if not labels:
            return item_id

        # only one label
        first_label = next(
            (
                label
                for label in labels
                if not (ENT_PATTERN.match(label) or PRE_PATTERN.match(label))
            ),
            labels[0],
        )
        return first_label
