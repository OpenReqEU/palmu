# palmu
UH Related similarity Detector for Qt Jira data

This service was created as a result of the OpenReq project funded by the European Union Horizon 2020 Research and Innovation programme under grant agreement No 732463.

# Technical Description

Palmu uses vectors to find similar issues within Qt Jira data.
The set of Qt issues is transformed to vector representations using FastText embeddings, and a fast similarity search algorithm is able to find the nearest neighbords for a given query. The idea is that vectors that are close in the embedding space must be somehow related and correspond to duplicates or dependencies in the _issue_ space. This search seems to be good to reduce the search space ( from hundred thousands to hundreds ) in the reduced space a random forest classifier is applied to output the _k-th_ most likely dependencies. 

## The following technologies are used:
- Python
- Docker
- faiss
	
## Public APIs

The service has not been deployed yet. Thus, there's no public API available.

## How to Install

Must have valid project requirement JSON files in the /data/ folder for the program to build.
Then, with Docker installed, run (this will take a while)

docker build . -t palmu

then

docker run -p 9210:9210 palmu

## How to Use This Microservice

GET hostname:9210/getRelated?id={issueId}k={}

Returns a String list of  _k_ closest related issues to the given issueId  (requires projects posted)

POST hostname:9210/postProject

(project JSON in request body)

Post a new project to Palmu

POST hostname:9210/newIssue 

valid OpenReq JSON must be in the request. The system will add this new data point to the current database and then perfom the search. 



## Notes for Developers

None at the moment.

## Sources

None

# How to Contribute
See the OpenReq Contribution Guidelines [here](https://github.com/OpenReqEU/OpenReq/blob/master/CONTRIBUTING.md).

# License

Free use of this software is granted under the terms of the [EPL version 2 (EPL2.0)](https://www.eclipse.org/legal/epl-2.0/).

