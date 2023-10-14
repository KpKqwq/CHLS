from reader import Reader_lexical
from metrics.evaluation import evaluation

def write_all_results(main_word, instance, target_pos, output_results, substitutes, substitutes_scores,
                      evaluation_metric):
    proposed_words = {}
    for substitute_str, score in zip(substitutes, substitutes_scores):
        # substitute_lemma = lemma_word(
        #     substitute_str,
        #     target_pos
        # ).lower().strip()

        substitute_lemma=substitute_str.lower().strip()
        max_score = proposed_words.get(substitute_lemma)
        if max_score is None or score > max_score:
            #if pos_filter(pos_vocab,target_pos,substitute_str,substitute_lemma):
            proposed_words[substitute_lemma] = score

    evaluation_metric.write_results(
        output_results + "_probabilites.txt",
        main_word, instance,
        proposed_words
    )
    evaluation_metric.write_results_p1(
        output_results + "_p1.txt",
        main_word, instance,
        proposed_words
    )

    evaluation_metric.write_results_p1(
        output_results + "_p3.txt",
        main_word, instance,
        proposed_words,limit=3
    )

    evaluation_metric.write_results_lex_oot(
        output_results + ".oot",
        main_word, instance,
        proposed_words, limit=50
    )

    # evaluation_metric.write_results_lex_oot(
    #     output_results + ".oot",
    #     main_word, instance,
    #     proposed_words, limit=10
    # )

    evaluation_metric.write_results_lex_best(
        output_results + ".best",
        main_word, instance,
        proposed_words, limit=1
    )


def real_cal_best_oot(work_dir,output_SR_file,output_score_file,embed_quto,evaluation_metric,test_golden_file):
    output_results=work_dir+output_SR_file+".embed."+str(embed_quto)
    results_file=work_dir+output_score_file+".embed."+str(embed_quto)
    evaluation_metric.calculation_perl(
        test_golden_file,
        output_results + ".best",
        output_results + ".oot",
        results_file + ".best",
        results_file + ".oot"
    )
    evaluation_metric.calculation_p1(
        test_golden_file,
        output_results + "_p1.txt",
        results_file + "_p1.txt"
    )
    
    evaluation_metric.calculation_p3(
        test_golden_file,
        output_results + "_p3.txt",
        results_file + "_p3.txt"
    )