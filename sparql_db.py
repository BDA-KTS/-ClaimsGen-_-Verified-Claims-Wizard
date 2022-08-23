from SPARQLWrapper import SPARQLWrapper, JSON

# Specify the DBPedia endpoint
sparql = SPARQLWrapper("http://dbpedia.org/sparql")

# Below we SELECT both the hot sauce items & their labels
# in the WHERE clause we specify that we want labels as well as items
wiki_external_ref = "76972"
print(type(wiki_external_ref))
query = """
PREFIX wd: <http://www.wikidata.org/entity/>
SELECT DISTINCT ?dbpedia_id WHERE {
    ?dbpedia_id owl:sameAs ?wikidata_id  .
    ?dbpedia_id dbo:wikiPageID ?wikipedia_id .
    VALUES (?wikipedia_id) {(%s)}
}
"""
#print(query)
query = query % wiki_external_ref
#print(query)
sparql.setQuery(query)
sparql.setReturnFormat(JSON)
result = sparql.query().convert()

# The return data contains "bindings" (a list of dictionaries)
#print(result)

import pandas as pd

results_df = pd.json_normalize(result['results']['bindings'])
#print(results_df)
entity_uri = results_df.iloc[0]['dbpedia_id.value']
print(entity_uri)


