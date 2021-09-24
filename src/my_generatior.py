from src.my_preprocess import *
import tqdm
from src.my_metrics import *

raw_text = open("./data/divina_textonly.txt", "rb").read().decode(encoding="utf-8")

result = "|Nel |mez|zo |del |cam|min |di |no|stra |vi|ta  \
         |mi |ri|tro|vai |per |u|na |sel|va o|scu|ra, \
         |ché |la |di|rit|ta |via |e|ra |smar|ri|ta. \
         |E |quel|la |che |l’ a|ni|ma |di |Dio |cu|ra, \
|del |mio |a|mor, |che |mi |fa |di|scer|ne\
|di |quel|la |par|te |che |di |là |m’ ap|pu|ra.\
|E |io |a|vea |la |vir|tù |che |l’ u|der|ne\
|la |pri|ma |mi|se|ria |che |la |co|sa |giu|sti\
|a |la |co|da |sua |per|cuo|ta |la |ger|ne.\
|E |que|sta |lin|gua, |che |l’ un |po|co a|gu|sti\
|più |che |po|tea |le |sue |ma|ni |di|scer|ne,\
|co|me |sa|reb|be |lu|ce |si |ri|ciu|sti,\
|se |non |po|tea |me|mo|ria |le|va|ter|ne,\
|che |la |co|sa e |di |san|za |com’ |io |sce|sa\
|che |non |si |può |sì |la |ve|ra|ce |ster|ne,\
|e |l’ al|tra |mer|ta|re |sì |se|gue |stre|sa.\
|E |se |la |vo|glia |che |la |par|te |guer|ne\
|di |quel|la |sua |pa|ro|la |che |li |vol|se,\
|che |non |si |fa |da |l’ al|tra |suo |con|ver1|ne,\
|e |per|ché ’l |sol |che |la |pa|ro|la |scol|se\
|poi |che |la |mia |men|te |lui |si |ri|co|ne,\
|e |poi |di|cea:« |Quan|to |la |vi|sta |fol|se\
|co|me |suo |pa|dre |che |si |ca|gion |fo|re,\
|se |tu |voi |ch’ io |per |la |via |do|ve |scol|se».\
|E |io, |a |dun|que |l’ o|ra|bil |che |po|re\
|co|me |fos|se |pian|ger, |che |mi |ri|spuo|se\
|de |la |col|pa |de |la |mia |ma|ra|vo|re.\
|La |vir|tù |che |mi |sem|pre |si |ri|spo|se\
|di |quel |che |si |fa |che ’l |cor|po |di|scen|de\
|per |lo |pec|ca|to |che |l’ un |di |co|stro|se,\
|per |lo |spe|re |mol|to |si |ri|ter|nen|de\
|d’ in|con|tra|ta |con |la |sua |pa|ro|la,\
|che |si |di|sten|de |la |ma|dre |si |cen|de."

original_text = preprocess_text(raw_text, end_of_verse='\n', end_of_tercet='', start_of_verse='', word_level=True)
original_text = re.sub(r' <SEP> ', ' ', original_text)
# print(original_text)

word_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='\n-:,?“‘)—»«!”(";.’ ', lower=False)
word_tokenizer.fit_on_texts([raw_text])
real_words = set(word_tokenizer.word_index.keys())


def generation_metrics(result):
    result_verses = result.split("\n")

    avg_syll = average_syllables(result_verses)
    hend_ratio = correct_hendecasyllables_ratio(result_verses)
    # avg_syll = 0
    # hend_ratio = 0

    result_verses = re.sub(r'\|', '', result)
    result_verses = remove_punctuation(result_verses)

    plagiarism = ngrams_plagiarism(result_verses, original_text)

    gen_tokenizer = tfds.deprecated.text.Tokenizer()
    gen_words = gen_tokenizer.tokenize(result_verses)

    correctness, _ = correct_words_ratio(gen_words, real_words, return_errors=True)
    incorrectness_score = incorrectness(set(gen_words), real_words)

    result_verses = result_verses.split('\n')
    rhyme_ratio = 0  # chained_rhymes_ratio(result_verses)

    return avg_syll, hend_ratio, rhyme_ratio, plagiarism, correctness, incorrectness_score


avg_syll, hend_ratio, rhyme_ratio, plagiarism, correctness, incorrectness_score = generation_metrics(result)
print(avg_syll)
print(hend_ratio)
print(rhyme_ratio)
print(plagiarism)
print(correctness)
print(incorrectness_score)
