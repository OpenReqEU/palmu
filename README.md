# palmu
UH Related Issue Detector for Qt Jira data

This service was created as a result of the OpenReq project funded by the European Union Horizon 2020 Research and Innovation programme under grant agreement No 732463.

# Technical Description

Palmu uses vectors to find similar issues within Qt Jira data.

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

GET hostname:9210/get-related?id={issueId}

Returns a String list of closest related issue IDs (requires projects posted)

POST hostname:9210/post-project

(project JSON in request body)

Post a new project to Palmu


## Notes for Developers

None at the moment.

## Sources

None

# How to Contribute
See the OpenReq Contribution Guidelines [here](https://github.com/OpenReqEU/OpenReq/blob/master/CONTRIBUTING.md).

# License

Free use of this software is granted under the terms of the [EPL version 2 (EPL2.0)](https://www.eclipse.org/legal/epl-2.0/).

