import wikipedia
import codecs
import os
import re

data_location="Corpus"
concept=wikipedia.page("Software Project Management")

content=concept.content
file_name="content"
print(content)
concepts=["Software Project Management","Software testing","software development life cycle","Software release life cycle","software project reports","Software documentation","Software bug","Software requirements specification"]
#concepts=["Software bug","Software requirements specification"]
concepts= concepts.__add__ (["management information system","Enterprise resource planning","Information Technology","Business analyst","customer relationship management","supply chain management"])
concepts=concepts.__add__(["Software Design and Development","Software test documentation","Software quality assurance","Software architecture","Software architecture analysis method","Software versioning","Component-based software engineering","Software design pattern","Supply chain management software","Supply chain risk management","Software development effort estimation"])

print("Total concepts:",concepts.__len__())


#concepts=(["Software Design and Development","Software test documentation","Software quality assurance","Software architecture","Software architecture analysis method","Software versioning","Component-based software engineering","Software design pattern","Supply chain management software","Supply chain risk management","Software development effort estimation"])

for concept in concepts:
    try:
        page_info = wikipedia.page(concept)
        concept = re.sub('[^A-Za-z0-9 ]+', '', concept)
        file_name = os.path.join(data_location, "--"+concept+"--"+ '.txt')
        file_wiki = codecs.open(file_name, "w", "utf-8")
        print(page_info.content)
        txt = re.sub('[^A-Za-z0-9 ]+', '', page_info.content)
        txt=page_info.content
        print(txt)
        file_wiki.write(txt)
        file_wiki.close()

        for link in page_info.links[:40]:
            print("----------------------------------------------")
            connect = wikipedia.page(link)
            link=re.sub('[^A-Za-z0-9]+', '', link)
            file_name = os.path.join(data_location, concept+"_"+link + '.txt')
            file_wiki = codecs.open(file_name, "w", "utf-8")
            print(connect.content)
            #txt = re.sub('[^A-Za-z0-9. ]+', '', connect.content)
            txt=connect.content
            print(txt)
            file_wiki.write(txt)
            file_wiki.close()
    except wikipedia.exceptions.DisambiguationError as de:
        print concept + ':disambiguation error.'
        continue
    except Exception as e:
        print("Exception found"+str(e.message))
        continue



