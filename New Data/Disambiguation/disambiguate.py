usage='''
  Given as command line arguments
  (1) wikidataLinks.tsv 
  (2) wikidataLabels.tsv
  (optional 2') wikidataDates.tsv
  (3) wikipedia-ambiguous.txt
  (4) the output filename'''
'''writes lines of the form
        title TAB entity
  where <title> is the title of the ambiguous
  Wikipedia article, and <entity> is the 
  wikidata entity that this article belongs to. 
  It is OK to skip articles (do not output
  anything in that case). 
  (Public skeleton code)'''

import sys
import re
from nltk.corpus import stopwords
from parser_suchanek import Parser
from simpleKB import SimpleKB

wikidata = None
if __name__ == "__main__":
    if len(sys.argv) is 5:
        dateFile = None
        wikipediaFile = sys.argv[3]
        outputFile = sys.argv[4]
    elif len(sys.argv) is 6:
        dateFile = sys.argv[3]
        wikipediaFile = sys.argv[4]
        outputFile = sys.argv[5]
    else:
        print(usage, file=sys.stderr)
        sys.exit(1)

    wikidata = SimpleKB(sys.argv[1], sys.argv[2], dateFile)
    
# wikidata is here an object containing 4 dictionaries:
## wikidata.links is a dictionary of type: entity -> set(entity).
##                It represents all the entities connected to a
##                given entity in the yago graph
## wikidata.labels is a dictionary of type: entity -> set(label).
##                It represents all the labels an entity can have.
## wikidata.rlabels is a dictionary of type: label -> set(entity).
##                It represents all the entities sharing a same label.
## wikidata.dates is a dictionnary of type: entity -> set(date).
##                It represents all the dates associated to an entity.

# Note that the class Page has a method Page.label(),
# which retrieves the human-readable label of the title of an 
# ambiguous Wikipedia page.

    with open(outputFile, 'w', encoding="utf-8") as output:
        for page in Parser(wikipediaFile):
            # DO NOT MODIFY THE CODE ABOVE THIS POINT
            # or you may not be evaluated (you can add imports).
            # YOUR CODE GOES HERE:
            # NOT TODO: add stemming for links (stemming has dramatically increased the runtime, but no significant improvement)
            #TODO: delete the ones more general, i.e. with less information in rlabels
            res = ""
            maxScore = 0
            stop_words = set(stopwords.words('english'))
            # the weight for a match of data, label
            # or a overlap word in link
            weightDate = 20
            weightLabel = 10
            weightRlabel= 30
            weightLink = 1

            # get the year if there is a data in the content, the month and the day will be ignored
            # I will only take the first one matched because yagoDates.tsv is not complete
            match = re.search(r'\d{4}', page.content)
            if(match):
                year = match.group(0)
            else:
                year = ""

            #get all possible Wikidata entries
            rlabels = wikidata.rlabels[page.label()]
            for rlabel in rlabels:
                score = 0

            # check the rlabel
                rlabelSet = set(re.sub('[!@#$,.\"()<>]', '', rlabel).split("_"))
                contentSet = set(re.sub('[!@#$,.\"()]', '', page.content).split(" "))
                nameSet = set(page.label().split(" "))
                intersectR = rlabelSet.intersection(contentSet)
                intersectN = nameSet.intersection(contentSet)
                for x in intersectR:
                    if x not in intersectN and x not in stop_words:
                        score+=weightRlabel


            # check the wikidata entry labels
                ''' a more efficient method, but less robust
                labelNotMatched = True
                labels = wikidata.labels[rlabel]
                for label in labels:
                    if label in page.content and label != page.label():
                        score += weightLabel
                        labelNotMatched = False
                    #special case for name of locations
                    elif "," in label:
                        label = label.split(",")
                        for place in label:
                            if place in page.content and place != page.label() and place not in stop_words:
                                score += weightLabel
                                labelNotMatched = False
                # if non of the labels of this entry can fit with the content,
                # this entry is probably not the one we want, so we give a -1 penality
                if labelNotMatched:
                    score -= weightLabel/2
                '''
                labels = wikidata.labels[rlabel]
                labelOverlap = []
                for label in labels:
                    labelsSet = set(re.sub('[!@#$,.\"()<>]', '', label).split(" "))
                    labelIntersect = labelsSet.intersection(contentSet)
                    if labelIntersect:
                        for x in labelIntersect:
                            if x not in labelOverlap and x not in stop_words:
                                labelOverlap.append(x)
                    '''
                    if ',' in label:
                        labelImportant = label.split(",")[1]
                        if labelImportant in page.content and x not in stop_words:
                            score += 5*weightLabel
                     '''
                score += len(labelOverlap)*weightLabel

            # check the year
                if rlabel in wikidata.dates and year != "":
                    dates = wikidata.dates[rlabel]
                    for date in dates:
                        if year in date:
                            score += weightDate

            #check the links
                if rlabel in wikidata.links:
                    links = wikidata.links[rlabel]
                    overlap = []
                    for link in links:
                        linkSet = set(re.sub('[!@#$,.\"()<>]', '', link).split("_"))
                        intersect = linkSet.intersection(contentSet)
                        if len(intersect) > 0 and type(intersect) is not bool:
                            for x in intersect:
                                if x not in overlap and x not in stop_words:
                                    overlap.append(x)
                    score += len(overlap)

            #comparison and selection
                if score > maxScore:
                    maxScore = score
                    res = rlabel

            print(page.title, res, sep='\t', file=output)

