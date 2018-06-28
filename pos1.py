# -*- coding: utf-8 -*-

import json
import re
import nltk
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def tag_text_with_skills(text):
    sl = open('skillsList.txt').read()
    skill_list = sl.split('\n')
    skill_list1 = []
    for s in skill_list:
        if s.strip() != '':
            skill_list1.append(s.strip())

    skill_dict = {'uni':{}, 'bi':{}}

    for skill in skill_list1:
        if len(skill.split(' ')) == 1:
            skill_dict['uni'][skill.lower()] = skill
        elif len(skill.split(' ')) == 2:
            skill_dict['bi'][skill.lower()] = skill.replace(' ', '_')

    #print skill_dict

    cleanText = re.sub('\W+','', text)
    grams = []
    unigram_arr = text.split()
    bigram_arr = list(nltk.bigrams(text.split()))

    for u in unigram_arr:
        grams.append(u)
    for b in bigram_arr:
        grams.append(' '.join(b))
        #print grams


    for g in grams:
        if g.lower() in skill_dict['bi']:
            bg_skill = skill_dict['bi'][g.lower()]
            text = text.replace(g, bg_skill + '__true')

        if g.lower() in skill_dict['uni']:
            text = text.replace(g, skill_dict['uni'][g.lower()]+ '__true')

    sent_final = []

    for word in text.split():
        if '__true' in word:
            sent_final.append((word.replace('__true', ''),1))
        else:
            sent_final.append((word, 0))
    return sent_final

if __name__ == "__main__":
    jd_dump = open('jd.dump').read()
    jdd_arr = jd_dump.split('\n')
    jdd_final = []
    for j in jdd_arr:
        for ji in j.split('.'):
            jdd_final.append(ji)
    tagged_sentences = []
    for y in jdd_final:
        ts = tag_text_with_skills(y)
        for aa, bb in ts:
            if bb == 1:
                tagged_sentences.append(ts)
                #print ts
                break
#print outfile

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    #is_skill = True if sentence[index][1]==1 else False
    return {
        #'is_skill': is_skill,
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'prev2_word': '' if index < 2 else sentence[index - 2],
        'prev3_word': '' if index < 3 else sentence[index - 3],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'next1_word': '' if index > len(sentence) - 3 else sentence[index + 2],
        'next2_word': '' if index > len(sentence) - 4 else sentence[index + 3],
        'has_underscore': '_' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:],

    }

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y

cutoff = int(.75 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]

X, y = transform_to_dataset(training_sentences)

print len(training_sentences)
print len(test_sentences)

clf = Pipeline([
('vectorizer', DictVectorizer(sparse=False)),
('classifier', DecisionTreeClassifier())
])
clf.fit(X, y)
X_test, y_test = transform_to_dataset(test_sentences)
print "Accuracy:", clf.score(X_test, y_test)