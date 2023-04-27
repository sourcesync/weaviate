package main

import (
	"encoding/json"
	"fmt"
	"github.com/google/uuid"
	"io"
	"log"
	"net/http"
)

const (
	PORT = uint(7760)
)

var dataset_id = ""
var path = ""

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

func handleTrainStatus(w http.ResponseWriter, r *http.Request) {
	fmt.Println("DOING GET TRAIN STATUS")
	if r.Method != "GET" {
		http.Error(w, "Method is not suppored.", http.StatusNotFound)
	}
	reqBody, _ := io.ReadAll(r.Body)
	reqData := map[string]interface{}{}
	juErr := json.Unmarshal(reqBody, &reqData)
	if juErr != nil {
		log.Fatal(juErr, "could not unmarshal request body")
	}
	fmt.Println("\nGET REQUEST")
	values := map[string]interface{}{
		"datasetStatus": "completed",
	}
	jsonret, err := json.Marshal(values)
	if err != nil {
		log.Fatal(err)
	}
	w.Write(jsonret)
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
	fmt.Println("\nLOAD DATASET:")
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
	fmt.Println("\nDOING SEARCH, reqData:", reqData)
	// dim := reqData["topk"].(float64)
	dist := map[string]interface{}{
		"hello": "world",
	}
	search := float64(1)
	values := map[string]interface{}{
		"distance": dist,
		"indices":  dist,
		"search":   search,
	}
	fmt.Println(len(values))
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
	reqBody, _ := io.ReadAll(r.Body)
	reqData := map[string]interface{}{}
	juErr := json.Unmarshal(reqBody, &reqData)
	if juErr != nil {
		log.Fatal(juErr, "could not unmarshal request body")
	}
	fmt.Println("\nDELETING QUERIES")
	values := map[string]interface{}{
		"status": "ok",
	}
	jsonret, _ := json.Marshal(values)
	w.Write(jsonret)

}

func handleDeleteDataset(w http.ResponseWriter, r *http.Request) {
	reqBody, _ := io.ReadAll(r.Body)
	reqData := map[string]interface{}{}
	juErr := json.Unmarshal(reqBody, &reqData)
	if juErr != nil {
		log.Fatal(juErr, "could not unmarshal request body")
	}
	fmt.Println("\nDELETING DATASET")
	values := map[string]interface{}{
		"status": "ok",
	}
	jsonret, _ := json.Marshal(values)
	w.Write(jsonret)
}

func main() {
	http.HandleFunc("/v1.0/dataset/import", handleImportDataset)
	http.HandleFunc("/v1.0/dataset/train/status/*", handleTrainStatus)
	http.HandleFunc("/v1.0/dataset/load", handleLoadDataset)
	http.HandleFunc("/v1.0/demo/query/import", handleImportQueries)
	http.HandleFunc("/v1.0/dataset/focus", handleFocusDataset)
	http.HandleFunc("/v1.0/dataset/search", handleSearch)
	http.HandleFunc("/v1.0/dataset/unload", handleUnloadDataset)
	http.HandleFunc("/v1.0/dataset/remove/*", handleDeleteDataset)
	http.HandleFunc("/v1.0/demo/query/remove/*", handleDeleteQueries)

	fmt.Printf("Starting server at port %d\n", PORT)
	if err := http.ListenAndServe(fmt.Sprintf(":%d", PORT), nil); err != nil {
		log.Fatal(err)
	}

}
