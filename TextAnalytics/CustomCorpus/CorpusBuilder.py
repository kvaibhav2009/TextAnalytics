
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import pandas as pd
import codecs
import wikipedia
import warnings


data_location="/wiki_data/"
working_concepts = pd.read_excel('key_concepts.xlsx', sheetname='SBI Working')
wiki_topics = working_concepts['Working Concepts'].values
traverse_links = working_concepts['Traverse_links'].values
print 'Total', len(wiki_topics), "Wiki topics to retrieve!"

for i, main_topic in enumerate(wiki_topics):
    file_name = os.path.join(data_location, main_topic + '.txt' )
    file_present = os.path.isfile(file_name)
    try:
        if file_present:
            print i, 'reading...', file_name, traverse_links[i] == True
            file_wiki = codecs.open(file_name, "r", "utf-8")
            content = file_wiki.read()
            file_wiki.close()
        else:
            print i, 'loading...', main_topic, traverse_links[i] == True
            page = wikipedia.page(main_topic)
            content = page.content
            links = page.links[:30]

            file_wiki = codecs.open(file_name, "w", "utf-8")
            file_wiki.write(content)
            file_wiki.close()

            if traverse_links[i] == False:
                continue
            print main_topic, 'has', len(links), 'links'
            for link in links:
                file_name = os.path.join(data_location, link + '.txt' )
                file_present = os.path.isfile(file_name)
                if file_present:
                    print link + ' is already extracted'
                    continue
                else:
                    try:
                        print 'Extracting: ', main_topic, '>', link
                        page = wikipedia.page(link)
                        content = page.content
                        file_wiki = codecs.open(file_name, "w", "utf-8")
                        file_wiki.write(content)
                        file_wiki.close()
                    except wikipedia.exceptions.PageError as pe:
                        #print pe
                        print link + ':page error.'
                        continue
                    except wikipedia.exceptions.DisambiguationError as de:
                        #print de
                        print link + ':disambiguation error.'
                        continue
                    except Exception as e:
                        #print e
                        print link + "subtopic's link could not be retrieved!"
                        continue
    except Exception as e:
        print e
        print main_topic, "subtopic could not be retrieved!"
        continue


# print
# print 'Total ', len(list_articles), 'links collected!'
# meta_file = codecs.open("../data/wiki_metadata.csv", "w", "utf-8")
# meta_file.write(topic_list)
# meta_file.close()