
# GSI Technology's Gemini Plugin

The Gemini Plugin provides an alternative to Weaviate's native HNSW ANN implementation:
* It serves as a bridge between Weaviate and GSI Technology's Fast Vector Search (FVS)
* FVS provides efficient hardware accelerated vector search and is suitable for large scale datasets

# Architecture

All of the code for the Gemini supprt lives in two primary places:
* In the core Weaviate code-base including:
  * A "gemini" entity which sits alongside the native HNSW entity, living under "entities/vectorIndex"
  * A "gemini" index stub which sits alongside the native HNSW index code under "adapters/repos/db/index"
* In this Golang "module" directory, which includes:
  * An FVS REST API wrapper written in pure Golang
  * Golang code which implements the index "interface" that sits right behind the "gemini" index stub described above.
  * The main Weaviate codebase uses Golang's import mechanism to include this code.
  * Please note that this module may end up in its own separately named repository in the future.  This will depend on conversations with the main Weaviate developers during initial integration.

# Testing

We've provide unit test and code coverage tests for this module.

Please follow these instructions to reproduce them.

* Install golang version 1.20.1 or higher.  For Linux Ubuntu, we recommend you download "https://go.dev/dl/go1.20.1.linux-amd64.tar.gz" and follow the installation steps here: https://tecadmin.net/how-to-install-go-on-ubuntu-20-04/
* Run the script in this directory called "test.sh"


