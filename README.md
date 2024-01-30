# Description
The Generator is a part of ClaimsKG pipeline . The entire pipeline of ClaimsKG consists of two major building blocks, namely the Extractor and Generator. The Generator performs: i) entity annotation and linking ii) rating normalization, and iii) lifting and serialization. The input to the generator should be a file containing claims and their related metadata and the output is a Knowledge Graph built according to ClaimsKG Schema.

#Entity Annotation and Linking :
This module performs Named Entity Recognition and Disambiguation  (NERD) of the claims and their reviews.Python Entity Fishing Client is used in the latest release which dissambiguates against Wikidata. We then use WikipediaExternalRefId to dissambiguate against DBPedia.

#Rating Normalization:
This module provides a normalized rating score for all claims in the dataset, alongside the original ratings. The claims are classified  into four categories TRUE, FALSE, MIXTURE, OTHER respectively indicated within ClaimsKG

#Lifting and Serialization:
This module uses Rdflib python library to create the model and an abstract RDF graph to then serialize it in one of the supported formats (TTL,n3, XML,nt, pretty-xml,trix, trig, and nquads). Unique URI identifiers are generated as UUIDs that are based on a one-way hash of key attributes for each
instance.

### Social Science Usecase
John is a social scientist studying about online discourse. He wants acess to false claims within a definite time period about the US Presidential Elections. He visits the MH to find this method that helps him to generate a Knowledge Graph from various fact-checked claims. He uses the search box on the top of the interface and types in Claims Generator or Knowledge Graph Generator. The search functionality of the MH shows him a list or related methods and tutorials that provides John with methods that can help him generate knowledge graphs which he can then querry and find all relevant claims and reuse for his study. 


As a social scientist Mary wants to investigate the impact of Gun laws on the society. She has a huge data dump of claims, but wants to search those pertaining to gun laws over the entire time period. She uses the search box to find methods related to claims.The search functionality of the MH shows her a list or related methods and tutorials related to claims that can help her generate a knowledge graph out of it. She then searches the Knowledge Graph regarding all claims related to the Gun laws and it brings her a list of all relevant claims, be it true, false, mixed or others which she can reuse for her study.


Lily is a researcher who wants to study the evolution of false claims related to Covid or coronavirus. She collects claims from a number of fact-checking websites but does not have an easy way to pick only those that are false and also related to Covid. She uses the search box in MH to find methods related to fact-checking.The search functionality of the MH shows her a list or related methods and tutorials related to Fact-checking that can help her generate a knowledge graph out of it.She generates the knowledge graph and runs a search querry to find all false claims related to covid or coronavirus in a very short time. The then collects those claims and uses it for her research.

### ClaimsKG pipeline

![ClaimsKG pipeline](claimskg_pipeline.PNG)

### Keywords
Claims, Fact-checking, Entity Linking

### Structure

ClaimsKG folder -Contains the entity annotation, RDF generation, and rating normalization module


docs folder - contains spynx documentation


export.py - The main file to run the project

### Environment SetUp
This program requires Python 3.x to run.

### Dependencies

To install the dependencies you may use: `pip3 install -r requirements.txt`

### Data Model

![](model.png)




### Usage
### Input data
The output of Extractor module is the input to the generator module in ClaimsKG.
Sample output for Extractor module can be found at https://git.gesis.org/bda/ClaimsKG
### Sample Input
![](input_sample.PNG)
### Sample Output
![](output_sample.PNG)
### How to Use
- For usage information you may use 
```shell
    python3 export.py -h
```
* The options are the following: 
  * `--input [file]` Indicated the location of the zip file generated by the fake new extractor (mandatory)
  * `--output [file]` Specifies the output file for the model (default: out.ttl)
  * `--format [format]` Specifies the format of the output serialization. You may use any of the supported formats in the `rdflib` package (xml', 'n3', 'turtle', 'nt', 'pretty-xml', 'trix', 'trig' and 'nquads'; default: turtle)
  * `--model-uri` The base URI of the model (by default `http://data.gesis.org/claimskg/public/`) 
  * `--resolve` Specifies whether to resolve the annotations to DBPedia URIs. If this option is activated, the resolution is performed through SPARQL queries to the official DBPedia endpoint, which requires you to have an active Internet connection. Additionally, you will need a running instance of `redis-server` as the results of the queries are cached to prevent unnecessary queries from being performed. 
  * `--threshold [float_value]` If `--resolve` is present, specifies the cutoff confidence threshold to include annotations as a mention. 
  * `--include-body` If `--include-body` is supplied, the body of the claim review is included in the `schema:ClaimReview` instances through the `schema:reviewBody` property.




## Contact
Susmita.Gangopadhyay@gesis.org

## Publication 
1. ClaimsKG: A knowledge graph of fact-checked claims (Tchechmedjiev, A., Fafalios, P., Boland, K., Gasquet, M., Zloch, M., Zapilko, B., ... & Todorov, K. (2019). ClaimsKG: A knowledge graph of fact-checked claims. In The Semantic Web–ISWC 2019: 18th International Semantic Web Conference, Auckland, New Zealand, October 26–30, 2019, Proceedings, Part II 18 (pp. 309-324). Springer International Publishing.)
2. Truth or dare: Investigating claims truthfulness with claimskg (Gangopadhyay, S., Boland, K., Dessí, D., Dietze, S., Fafalios, P., Tchechmedjiev, A., ... & Jabeen, H. (2023, May). Truth or dare: Investigating claims truthfulness with claimskg. In D2R2’23-Second International Workshop on Linked Data-driven Resilience Research (Vol. 3401).)
  
  
 
 
 
