from gensim.models import word2vec
import pandas as pd
import wikipedia

import urllib
r = urllib.urlopen('https://www.bankofamerica.com/deposits/manage/glossary.go').read()

from bs4 import BeautifulSoup
soup_am = BeautifulSoup(r)
#soup_naic = BeautifulSoup(open("../resources/Glossary of Insurance Terms_NAIC.html"), 'html.parser')

links = []
glossary_terms = []
descs = []
counter = 0


for strong_tag in soup_am.find_all('p'):
    if strong_tag.next_sibling is not None:
        print strong_tag.text, strong_tag.next_sibling
        try:
            art_list = wikipedia.search(strong_tag.text, results=1)  # search_n[i]
            link1 = ''
            if len(art_list) > 0:
                link1 = art_list[0]
            counter += 1
        except:
            link1 = ''
            print strong_tag.text, 'not retrieved!'

        glossary_terms.append(strong_tag.text)
        links.append(link1)
        descs.append(strong_tag.next_sibling)


print counter, 'links retrieved'
print len(glossary_terms), len(links), len(descs)
df_links = pd.DataFrame(columns=['Glossary Term', 'Wiki Link', 'Description'])
df_links['Glossary Term'] = glossary_terms
df_links['Wiki Link'] = links
df_links['Description'] = descs

df_links.to_csv('glossary_links.csv', index=False, encoding='utf-8')





