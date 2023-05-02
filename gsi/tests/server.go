//                           _       _
// __      _____  __ ___   ___  __ _| |_ ___
// \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
//  \ V  V /  __/ (_| |\ V /| | (_| | ||  __/
//   \_/\_/ \___|\__,_| \_/ |_|\__,_|\__\___|
//
//  Copyright © 2016 - 2023 Weaviate B.V. All rights reserved.
//
//  CONTACT: hello@weaviate.io
//

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/kshedden/gonpy"
)

const (
	PORT = uint(7760)
)

var dataset_id string
var path string
var bits float64

func handleImportDataset(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method is not suppported.", http.StatusNotFound)
		return
	}
	reqBody, _ := io.ReadAll(r.Body)
	reqData := map[string]interface{}{}
	juErr := json.Unmarshal(reqBody, &reqData)
	if juErr != nil {
		log.Fatal(juErr, "could not unmarshal request body")
	}
	fmt.Println("\nIMPORT DATASET")
	bits = reqData["nbits"].(float64)
	path = reqData["dsFilePath"].(string)
	types := [3]string{"flat", "cluster", "hnsw"}
	searchType := reqData["searchType"]
	valid := false
	for i := 0; i < len(types); i++ {
		if types[i] == searchType {
			valid = true
		}
	}
	if !valid {
		http.Error(w, "Not valid search type", 400)
	} else {
		dataset_id = uuid.New().String()
		values := map[string]interface{}{
			"datasetId": dataset_id,
		}
		jsonret, err := json.Marshal(values)
		if err != nil {
			log.Fatal(err, "failed to marshal values")
		}
		w.Write(jsonret)
	}
}

func handleTrainStatus(w http.ResponseWriter, r *http.Request) {
	fmt.Println("\nTRAIN STATUS")
	if r.Method != "GET" {
		http.Error(w, "Method is not suppored.", http.StatusNotFound)
	}
	reader, _ := gonpy.NewFileReader(path)
	if reader.Shape[0] < 4000 {
		values := map[string]interface{}{
			"datasetStatus": "error",
		}
		jsonret, _ := json.Marshal(values)
		w.Write(jsonret)
	} else {
		values := map[string]interface{}{
			"datasetStatus": "completed",
		}
		jsonret, err := json.Marshal(values)
		if err != nil {
			log.Fatal(err)
		}
		w.Write(jsonret)
	}
}

func handleLoadDataset(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method is not supported", http.StatusNotFound)
	}
	reqBody, _ := io.ReadAll(r.Body)
	reqData := map[string]interface{}{}
	juErr := json.Unmarshal(reqBody, &reqData)
	if juErr != nil {
		log.Fatal(juErr, "could not unmarshal request body")
	}
	fmt.Println("\nLOAD DATASET")
	if uint(bits)%2 != 0 {
		values := map[string]interface{}{
			"detail": "The server encountered an internal error and was unable to complete your request. Either the server is overloaded or there is an error in the application.",
			"status": 500,
			"title":  "Internal Server Error",
			"type":   "about:blank",
		}
		jsonret, _ := json.Marshal(values)
		w.Write(jsonret)
	} else {
		values := map[string]interface{}{
			"status": "ok",
			"title":  "none",
		}
		jsonret, err := json.Marshal(values)
		if err != nil {
			log.Fatal(err)
		}
		w.Write(jsonret)
	}
}

func handleImportQueries(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method is not supported.", http.StatusNotFound)
	}
	reqBody, _ := io.ReadAll(r.Body)
	reqData := map[string]interface{}{}
	juErr := json.Unmarshal(reqBody, &reqData)
	if juErr != nil {
		log.Fatal(juErr, "could not unmarshal request body")
	}
	fmt.Println("\nIMPORT QUERIES")
	qid := uuid.New().String()
	values := map[string]interface{}{
		"addedQuery": map[string]interface{}{
			"id": qid,
		},
	}
	jsonret, err := json.Marshal(values)
	if err != nil {
		log.Fatal(err)
	}
	w.Write(jsonret)
}

func handleFocusDataset(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method is not supported.", http.StatusNotFound)
	}
	reqBody, _ := io.ReadAll(r.Body)
	reqData := map[string]interface{}{}
	juErr := json.Unmarshal(reqBody, &reqData)
	if juErr != nil {
		log.Fatal(juErr, "could not unmarshal request body")
	}
	fmt.Println("\nFOCUS DATASET")
	values := map[string]interface{}{
		"hello": "world",
	}
	jsonret, err := json.Marshal(values)
	if err != nil {
		log.Fatal(err)
	}
	w.Write(jsonret)
}

func handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method is not supported", http.StatusNotFound)
	}
	reqBody, _ := io.ReadAll(r.Body)
	reqData := map[string]interface{}{}
	juErr := json.Unmarshal(reqBody, &reqData)
	if juErr != nil {
		log.Fatal(juErr, "could not unmarshal request body")
	}
	fmt.Println("\nSEARCH")
	dim := reqData["topk"].(float64)
	dist := make([][]float32, int(dim))
	for i := 0; i < len(dist); i++ {
		dist[i] = make([]float32, int(dim))
		for j := 0; j < len(dist); j++ {
			dist[i][j] = float32(1)
		}
	}
	search := float64(.001)
	values := map[string]interface{}{
		"distance": dist,
		"indices":  dist,
		"search":   search,
	}
	jsonret, _ := json.Marshal(values)
	w.Write(jsonret)
}

func handleUnloadDataset(w http.ResponseWriter, r *http.Request) {
	reqBody, _ := io.ReadAll(r.Body)
	reqData := map[string]interface{}{}
	juErr := json.Unmarshal(reqBody, &reqData)
	if juErr != nil {
		log.Fatal(juErr, "could not unmarshal request body")
	}
	fmt.Println("\nUNLOADING DATASET")
	values := map[string]interface{}{
		"status": "ok",
	}
	jsonret, _ := json.Marshal(values)
	w.Write(jsonret)

}

func handleDeleteQueries(w http.ResponseWriter, r *http.Request) {
	fmt.Println("\nDELETING QUERIES")
	values := map[string]interface{}{
		"status": "ok",
	}
	jsonret, _ := json.Marshal(values)
	w.Write(jsonret)

}

func handleDeleteDataset(w http.ResponseWriter, r *http.Request) {
	fmt.Println("\nDELETING DATASET")
	values := map[string]interface{}{
		"status": "ok",
	}
	jsonret, _ := json.Marshal(values)
	w.Write(jsonret)
}

func main() {

	myRouter := mux.NewRouter().StrictSlash(true)

	myRouter.HandleFunc("/v1.0/dataset/import", handleImportDataset)
	myRouter.HandleFunc("/v1.0/dataset/train/status/{dataset_id}", handleTrainStatus)
	myRouter.HandleFunc("/v1.0/dataset/load", handleLoadDataset)
	myRouter.HandleFunc("/v1.0/demo/query/import", handleImportQueries)
	myRouter.HandleFunc("/v1.0/dataset/focus", handleFocusDataset)
	myRouter.HandleFunc("/v1.0/dataset/search", handleSearch)
	myRouter.HandleFunc("/v1.0/dataset/unload", handleUnloadDataset)
	myRouter.HandleFunc("/v1.0/dataset/remove/{dataset_id}", handleDeleteDataset)
	myRouter.HandleFunc("/v1.0/demo/query/remove/{query_id}", handleDeleteQueries)

	fmt.Printf("Starting server at port %d\n", PORT)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", PORT), myRouter))
}
