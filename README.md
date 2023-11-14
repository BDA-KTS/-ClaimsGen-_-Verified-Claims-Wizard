# Description
The Generator is a part of ClaimsKG pipeline . The entire pipeline of ClaimsKG consists of two major building blocks, namely the Extractor and Generator. The Generator performs: i) entity annotation and linking ii) rating normalization, and iii) lifting and serialization. The input to the generator should be a file containing claims and their related metadata and the output is a Knowledge Graph.

#Entity Annotation and Linking :
This module performs Named Entity Recognition and Disambiguation  (NERD) of the claims and their reviews.Python Entity Fishing Client is used in the latest release which dissambiguates against Wikidata. We then use WikipediaExternalRefId to dissambiguate against DBPedia.

#Rating Normalization:
This module provides a normalized rating score for all claims in the dataset, alongside the original ratings. The claims are classified  into four categories TRUE, FALSE, MIXTURE, OTHER respectively indicated within ClaimsKG

#Lifting and Serialization:
This module uses Rdflib python library to create the model and an abstract RDF graph to then serialize it in one of the supported formats (TTL,n3, XML,nt, pretty-xml,trix, trig, and nquads). Unique URI identifiers are generated as UUIDs that are based on a one-way hash of key attributes for each
instance.

### ClaimsKG pipeline

![ClaimsKG pipeline](claimskg_pipeline.PNG)

### Structure

ClaimsKG-Contains the entity annotation, RDF generation, and rating normalization module


docs- contains spynx documentation


export.py - main file to run the project

### Data Model

![](model.png)

### Data

The output of Extractor module is the input to the generator module in ClaimsKG.
Sample output for Extractor module can be found at https://git.gesis.org/bda/ClaimsKG

### Installation & Requirements

This program requires Python 3.x to run.

To install the dependencies you may use: `pip3 install -r requirements.txt`

### Usage 
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
  
  
 
 
 
