import nltk
nltk.download('punkt')    
nltk.download('stopwords')

import math

from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords


with open('input.txt', 'r') as myfile:
  text_str = myfile.read()



def freqTableCreate(text_string):
    words = word_tokenize(text_string)
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def freqMatrixCreate(sentences):
    stopWords = set(stopwords.words("english"))
    frequency_matrix = {}
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


def calc_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


def calc_documents_per_word(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def calc_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def calc_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def sentence_sorting(tf_idf_matrix):

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


def avgScore(sentenceValue):
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average


def createSummary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text):
    
    
    # 1 Sentence Tokenize
    sentences = sent_tokenize(text)
    total_documents = len(sentences)
    #print(sentences)

    # 2 Create the Frequency matrix of the words in each sentence.
    freq_matrix = freqMatrixCreate(sentences)
    #print(freq_matrix)


    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = calc_tf_matrix(freq_matrix)
    #print(tf_matrix)

    # 4 creating table for documents per words
    count_doc_per_words = calc_documents_per_word(freq_matrix)
    #print(count_doc_per_words)


    # 5 Calculate IDF and generate a matrix
    idf_matrix = calc_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    #print(idf_matrix)

    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = calc_tf_idf_matrix(tf_matrix, idf_matrix)
    #print(tf_idf_matrix)

    # 7 Score the sentences
    sentence_scores = sentence_sorting(tf_idf_matrix)
    #print(sentence_scores)

    # 8 Find the threshold
    threshold = avgScore(sentence_scores)
    #print(threshold)

    # 9 Generate the summary
    summary = createSummary(sentences, sentence_scores,  1.1*threshold)     #multiply threshold with 0.0-infinity to change size of summary, default value is the avg value of all sentences
    
    return summary


if __name__ == '__main__':
    summary = run_summarization(text_str)
    print('\nSaving summary to file name "output.txt"\n')
    info="Summary info:\nLength of original text= "+str(len(text_str))+"\nLength of summary= "+str(len(summary))+"\n\nSummary:\n"
    with open('output.txt', 'w') as myfile:
        myfile.write(info)
        myfile.write(summary)
        myfile.flush()
        myfile.close()
    print('\nSummary saved\n')