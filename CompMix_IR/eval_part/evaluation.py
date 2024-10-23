import json

from tqdm import tqdm

from .library.string_library import StringLibrary
# from Levenshtein import distance as levenshtein_distance


def answer_presence(evidences, answers):
    """
    Compute the answer presence for a set of evidences
    and a parsed answer dict, and return a list of
    answering evidences.
    Return format: (boolean, [evidence-dict, ...])
    """
    # initialize
    answer_present = False
    answering_evidences = list()

    # go through evidences
    for evidence in evidences:
        if evidence_has_answer(evidence, answers):
            # remember evidence
            answer_present = True
            answering_evidences.append(evidence)
    # return results
    return (answer_present, answering_evidences)


def evidence_has_answer(evidence, gold_answers):
    """Check whether the given evidence has any of the answers."""
    for answer_candidate in evidence["wikidata_entities"]:
        # check if answering candidate
        if candidate_in_answers_id(answer_candidate, gold_answers) or candidate_in_answers_label(answer_candidate, gold_answers):
            return True
    
    for answer_candidate in evidence["disambiguations"]:
        ans_mention = answer_candidate[0]
        ans_id = answer_candidate[1]
        if ans_id is None or ans_id == False:
            continue
        if candidate_in_answers_disambiguations(ans_mention, gold_answers):
            return True

    return False


def candidate_in_answers_disambiguations(answer_candidate, gold_answers):
    """Check if candidate is answer."""
    # get ids
    # answer_candidate_id = answer_candidate["label"]
    gold_answer_ids = [answer["label"] for answer in gold_answers]

    # normalize
    answer_candidate = answer_candidate.lower().strip().replace('"', "").replace("+", "")
    gold_answer_ids = [answer.lower().strip().replace('"', "") for answer in gold_answer_ids]

    # perform check
    if answer_candidate in gold_answer_ids:
        return True

    # no match found
    return False



def candidate_in_answers_id(answer_candidate, gold_answers):
    """Check if candidate is answer."""
    # get ids
    answer_candidate_id = answer_candidate["id"]
    gold_answer_ids = [answer["id"] for answer in gold_answers]

    # normalize
    answer_candidate_id = answer_candidate_id.lower().strip().replace('"', "").replace("+", "")
    gold_answer_ids = [answer.lower().strip().replace('"', "") for answer in gold_answer_ids]

    # perform check
    if answer_candidate_id in gold_answer_ids:
        return True

    # no match found
    return False


def candidate_in_answers_label(answer_candidate, gold_answers):
    """Check if candidate is answer."""
    # get ids
    answer_candidate_id = answer_candidate["label"]
    gold_answer_ids = [answer["label"] for answer in gold_answers]

    # normalize
    answer_candidate_id = answer_candidate_id.lower().strip().replace('"', "").replace("+", "")
    gold_answer_ids = [answer.lower().strip().replace('"', "") for answer in gold_answer_ids]

    # perform check
    if answer_candidate_id in gold_answer_ids:
        return True

    # no match found
    return False


def mrr_score(evidences, gold_answers):
    """Compute MRR scores for given answers and gold answers at different thresholds."""
    def compute_mrr_at_k(evidences, gold_answers, k):
        for rank, evidence in enumerate(evidences[:k]):
            if evidence_has_answer(evidence, gold_answers):
                return 1.0 / float(rank + 1)
        return 0.0

    mrr_at_10 = compute_mrr_at_k(evidences, gold_answers, 10)
    mrr_at_20 = compute_mrr_at_k(evidences, gold_answers, 20)
    mrr_at_50 = compute_mrr_at_k(evidences, gold_answers, 50)
    mrr = compute_mrr_at_k(evidences, gold_answers, len(evidences))

    return mrr_at_10, mrr_at_20, mrr_at_50, mrr



def hit_at_k(evidences, gold_answers, k):
    """Compute Hit@k score for given answers and gold answers."""

    evidences = evidences[:k]
    hit = 0.0
    sum_hit = 0.0
    for evidence in evidences:
        if evidence_has_answer(evidence, gold_answers):
            hit = 1.0
            sum_hit += 1.0

    return hit, sum_hit


def hit_at_100(evidences, gold_answers):

    # assert len(evidences) == 100
    evidences = evidences[:100]
    
    hit_sum = []
    for evidence in evidences:
        if evidence_has_answer(evidence, gold_answers):
            hit_sum.append(1.0)
        else:
            hit_sum.append(0.0)

    # Create the prefix sum list
    prefix_sum = []
    current_sum = 0.0
    for value in hit_sum:
        current_sum += value
        prefix_sum.append(current_sum)

    return prefix_sum


def get_ranked_answers(config, generated_answer, turn):
    """
    Convert the predicted answer text to a Wikidata ID (or Yes/No),
    and return the ranked answers.
    Can be used for any method that predicts an answer string (instead of a KB item).
    """
    # check if existential (special treatment)
    question = turn["question"]
    if question_is_existential(question):
        ranked_answers = [
            {"answer": {"id": "yes", "label": "yes"}, "score": 1.0, "rank": 1},
            {"answer": {"id": "no", "label": "no"}, "score": 0.5, "rank": 2},
        ]
    # no existential
    else:
        # return dummy answer in case None was found (if no evidences found)
        if generated_answer is None:
            return [{"answer": {"id": "None", "label": "None"}, "rank": 1, "score": 0.0}]
        smallest_diff = 100000
        all_answers = list()
        mentions = set()
        for evidence in turn["top_evidences"]:
            for disambiguation in evidence["disambiguations"]:
                mention = disambiguation[0]
                id = disambiguation[1]
                if id is None or id == False:
                    continue

                # skip duplicates
                ans = str(mention) + str(id)
                if ans in mentions:
                    continue
                mentions.add(ans)
                # exact match
                if generated_answer == mention:
                    diff = 0
                # otherwise compute edit distance
                else:
                    diff = levenshtein_distance(generated_answer, mention)

                all_answers.append({"answer": {"id": id, "label": mention}, "score": diff})

        sorted_answers = sorted(all_answers, key = lambda j: j['score'])
        ranked_answers = [
            {"answer": answer["answer"], "score": answer["score"], "rank": i+1}
            for i, answer in enumerate(sorted_answers)
        ]

    # don't return all answers
    max_answers = config["ha_max_answers"]
    ranked_answers = ranked_answers[:max_answers]
    if not ranked_answers:
        ranked_answers = [{"answer": {"id": "None", "label": "None"}, "rank": 1, "score": 0.0}]
    return ranked_answers


def question_is_existential(question):
    existential_keywords = [
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "being",
        "been",
        "did",
        "do",
        "does",
        "done",
        "doing",
        "has",
        "have",
        "had",
        "having",
    ]
    lowercase_question = question.lower()
    lowercase_question = lowercase_question.strip()
    for keyword in existential_keywords:
        if lowercase_question.startswith(keyword):
            return True
    return False
