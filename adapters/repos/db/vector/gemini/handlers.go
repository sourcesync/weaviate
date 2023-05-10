package gemini

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/google/uuid"
	"github.com/kshedden/gonpy"
)

type myGemini struct {
	dataset_id       string
	bits             float64
	path             string
	allocation_token string
}

var server myGemini

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
	server.bits = reqData["nbits"].(float64)
	server.path = reqData["dsFilePath"].(string)
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
		dataset_id := uuid.New().String()
		server.dataset_id = dataset_id
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
	if r.Method != "GET" {
		http.Error(w, "Method is not suppored.", http.StatusNotFound)
	}
	reader, _ := gonpy.NewFileReader(server.path)
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
	if uint(server.bits)%2 != 0 {
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
	values := map[string]interface{}{
		"status": "ok",
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
	values := map[string]interface{}{
		"status": "ok",
	}
	jsonret, _ := json.Marshal(values)
	w.Write(jsonret)
}

func handleDeleteQueries(w http.ResponseWriter, r *http.Request) {
	values := map[string]interface{}{
		"status": "ok",
	}
	jsonret, _ := json.Marshal(values)
	w.Write(jsonret)
}

func handleDeleteDataset(w http.ResponseWriter, r *http.Request) {
	values := map[string]interface{}{
		"status": "ok",
	}
	jsonret, _ := json.Marshal(values)
	w.Write(jsonret)
}

func handleListDatasets(w http.ResponseWriter, r *http.Request) {
	jsonstr := `{
		"datasetsList": [
		  	{
				"datasetCopyFilePath": null,
				"datasetFileType": null,
				"datasetName": "generated_20230319161738",
				"datasetStatus": "completed",
				"datasetType": "float32",
				"encodingDetails": [
					{
						"binFilePath": null,
						"id": "9640463e-c671-11ed-93bb-0242ac1c000f",
						"isActive": true,
						"nbits": 768,
						"neuralMatrix": "/home/public/elastic-similarity/float32_neural/0b391a1a-b916-11ed-afcb-0242ac1c0002/datasets/23c5d805-cf99-4a77-a669-814c62c64887/weights/weights_768.h5",
						"neuralMatrixMetadata": "/home/public/elastic-similarity/float32_neural/0b391a1a-b916-11ed-afcb-0242ac1c0002/datasets/23c5d805-cf99-4a77-a669-814c62c64887/weights/weights_768_metadata.txt"
				  	}
				],
				"hamming": 3200,
				"id": "23c5d805-cf99-4a77-a669-814c62c64887",
				"metadataCopyFilePath": null,
				"metadataFilePath": null,
				"numOfFeatures": 960,
				"numOfRecords": 1000000,
				"searchType": "flat",
				"sysCreDate": "2023-03-19 16:17:39",
				"targetAccuracy": null,
				"transactionsFilePath": null,
				"uniqueMetadata": false
			},
			{
				"datasetCopyFilePath": null,
				"datasetFileType": null,
				"datasetName": "generated_20230319182701",
				"datasetStatus": "completed",
				"datasetType": "float32",
				"encodingDetails": [
					{
						"binDatasetSizeInBytes": 96000000,
						"binFilePath": "/home/public/elastic-similarity/float32_neural/0b391a1a-b916-11ed-afcb-0242ac1c0002/datasets/9b228c21-dee9-470c-ad2e-9908525d025f/binaries/20230319182848/records.npy",
						"id": "a4de585e-c683-11ed-93bb-0242ac1c000f",
						"isActive": true,
						"nbits": 768,
						"neuralMatrix": "/home/public/elastic-similarity/float32_neural/0b391a1a-b916-11ed-afcb-0242ac1c0002/datasets/9b228c21-dee9-470c-ad2e-9908525d025f/weights/weights_768.h5",
						"neuralMatrixMetadata": "/home/public/elastic-similarity/float32_neural/0b391a1a-b916-11ed-afcb-0242ac1c0002/datasets/9b228c21-dee9-470c-ad2e-9908525d025f/weights/weights_768_metadata.txt"
					}
				],
				"hamming": 1000,
				"id": "9b228c21-dee9-470c-ad2e-9908525d025f",
				"metadataCopyFilePath": null,
				"metadataFilePath": null,
				"numOfFeatures": 96,
				"numOfRecords": 1000000,
				"searchType": "flat",
				"sysCreDate": "2023-03-19 18:27:01",
				"targetAccuracy": 100,
				"transactionsFilePath": null,
				"uniqueMetadata": false
			}
		]
	}`
	w.Write([]byte(jsonstr))
}

func handleListLoaded(w http.ResponseWriter, r *http.Request) {
	server.allocation_token = r.Header.Get("allocationToken")
	jsonstr := fmt.Sprintf(`{
		"allocationsList": {
		  	"%s": {
				"allocationName": null,
				"datasetInFocus": {
				  	"datasetId": null,
			  		"neuralMatrixId": null
				},
				"loadedDatasets": [
					{
						"centroidsHammingK": 5000,
						"centroidsK": 4000,
						"datasetId": "%s",
						"hammingK": 3200,
						"isLoaded": true,
						"rerankTopK": 25,
						"searchType": "clusters"
					}
				],
				"maxNumOfThreads": 5,
				"numOfBoards": 1
			}
		}
	}`, server.allocation_token, server.dataset_id)
	w.Write([]byte(jsonstr))
}
