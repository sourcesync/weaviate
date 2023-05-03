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

package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/kshedden/gonpy"
	"github.com/stretchr/testify/assert"
	"golang.org/x/exp/mmap"
)

const (
	HOST  = "localhost"
	PORT  = uint(7760)
	ALLOC = "fd283b38-3e4a-11eb-a205-7085c2c5e516"
	// var ALLOC = "0b391a1a-b916-11ed-afcb-0242ac1c0002"
	VERBOSE = true
	FAKE    = true
)

var (
	topk        = uint(5)
	bits        = float64(128)
	search_type = "flat"
	path        = ""
	dataset_id  = ""
	query_id    = ""
)

// Generate a random string useful for generate a temp filename that does not already exist
func randomString(length int) string {
	rand.Seed(time.Now().UnixNano())
	b := make([]byte, length+2)
	rand.Read(b)
	return fmt.Sprintf("%x", b)[2 : length+2]
}

// This function is for testing all the FVS 'NumpyAppend*" functions
func TestFVSNumpyFunctions(t *testing.T) {
	// prepare for numpy float32 tests
	// get a temp file name
	ranstr := randomString(10)
	ranfilepath := fmt.Sprintf("/tmp/gemini_plugin_test_%s", ranstr)

	// Run all unit tests in parallel for speed
	t.Parallel()

	// Run a unit test for the function "Numpy_append_float32_array"
	t.Run("NumpyAppendFloat32", func(t *testing.T) {
		// make sure the file does not exist
		_, ferr := os.Stat(ranfilepath)
		// we are expecting an err because the file should not exist
		assert.NotNilf(t, ferr, "The file already exists.")

		// create a float array of dims = 96
		arr := make([][]float32, 1)
		arr[0] = make([]float32, 96)

		//
		// Execute the function
		//
		row_count, dim, aerr := Numpy_append_float32_array(ranfilepath, arr, 96, 1)

		//
		// Perform various checks on expected return values
		//

		// we are expecting nil for aerr
		assert.Nilf(t, aerr, "Got error for Numpy_append_float32_array")

		// we are expecting a specific "row_count" value
		assert.Equal(t, 1, row_count, "Expecting row_count of 1")

		// we are expecting a specific "dim" value
		assert.Equal(t, 96, dim, "Expecting dim of 96")
	})

	t.Run("NumpyReadFloat32", func(t *testing.T) {
		// Start float32 read test
		readerAt, ferr := mmap.Open(ranfilepath)

		// expecting nil for ferr
		assert.Nilf(t, ferr, "Got error for opening numpy file")

		// create a float array of dims = 96
		arr := make([][]float32, 1)
		arr[0] = make([]float32, 96)

		// read numpy file and store dims
		rdim, aerr := Numpy_read_float32_array(readerAt, arr, int64(96), int64(0), int64(1), int64(128))

		// nil for aerr
		assert.Nilf(t, aerr, "Got error for Numpy_read_float32_array")

		// 96 for dim
		assert.Equal(t, int64(96), rdim, "Expecting dims of 96")

		//
		// Unit test cleanup
		//
		derr := os.Remove(ranfilepath)
		assert.Nilf(t, derr, "Could not delete the temp file")
	})

	// prepare for numpy uint32 tests
	// get a temp file name
	ranstr = randomString(10)
	ranfilepath = fmt.Sprintf("/tmp/gemini_plugin_test_%s", ranstr)

	// Run a unit test for the function "Numpy_append_uint32_array"
	t.Run("NumpyAppendUint32", func(t *testing.T) {
		// make sure file does not exist
		_, ferr := os.Stat(ranfilepath)
		// expecting an err because file should not exist
		assert.NotNilf(t, ferr, "The file alerady exists")

		// create uint array
		arr := make([][]uint32, 1)
		arr[0] = make([]uint32, 96)

		// write uint32 numpy file
		aerr := Numpy_append_uint32_array(ranfilepath, arr, 96, 1)
		// expecting nil for aerr
		assert.Nilf(t, aerr, "Got error for Numpy_append_uint32_array")
	})

	t.Run("NumpyReadUint32", func(t *testing.T) {
		// open numpy file
		readerAt, err := mmap.Open(ranfilepath)
		// expecting nil for err
		assert.Nilf(t, err, "Could not open numpy file")

		// create a uint32 array
		arr := make([][]uint32, 1)
		arr[0] = make([]uint32, 96)

		// read from numpy file
		dim, aerr := Numpy_read_uint32_array(readerAt, arr, int64(96), int64(0), int64(1), int64(128))
		// expecting nil for aerr
		assert.Nilf(t, aerr, "Could not read from numpy file")
		// expecting 96 for dim
		assert.Equal(t, int64(96), dim, "Expecting dims of 96")

		// all tests passed, cleanup
		derr := os.Remove(ranfilepath)
		assert.Nilf(t, derr, "Could not delete the temp file")
	})
}

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
	fmt.Println("dataset shape:", reader.Shape)
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

func TestFakeServer(t *testing.T) {
	if FAKE == false {
		t.Skip("using Gemini FVS hardware for server")
	} else {
		fmt.Println("Using fake server instead of Gemini hardware")
	}

	router := mux.NewRouter().StrictSlash(true)
	router.HandleFunc("/v1.0/dataset/import", handleImportDataset)
	router.HandleFunc("/v1.0/dataset/train/status/{dataset_id}", handleTrainStatus)
	router.HandleFunc("/v1.0/dataset/load", handleLoadDataset)
	router.HandleFunc("/v1.0/demo/query/import", handleImportQueries)
	router.HandleFunc("/v1.0/dataset/focus", handleFocusDataset)
	router.HandleFunc("/v1.0/dataset/search", handleSearch)
	router.HandleFunc("/v1.0/dataset/unload", handleUnloadDataset)
	router.HandleFunc("/v1.0/dataset/remove/{dataset_id}", handleDeleteDataset)
	router.HandleFunc("/v1.0/demo/query/remove/{query_id}", handleDeleteQueries)

	ctx := context.Background()
	graceperiod := 5 * time.Second
	httpAddr := ":7760"

	srv := &http.Server{
		Addr:    httpAddr,
		Handler: router,
	}

	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGINT, syscall.SIGTERM)

	fmt.Println("starting http server")
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()

	t.Parallel()
	ctxTimeout, cancel := context.WithTimeout(ctx, graceperiod)
	defer func() {
		cancel()
		fmt.Println("STOPPING SERVER")
	}()

	if err := srv.Shutdown(ctxTimeout); err != nil {
		log.Fatal(err)
	}
}

// Tests for FVS functions
// Test1: 10k dataset, 10 queries, 128 bits, flat search, topk 5
// no errors expected
func TestFVSFunctions1(t *testing.T) {
	// setup for FVS testing
	fmt.Println("----------TEST 1----------")
	ranstr := randomString(10)
	path := fmt.Sprintf("/tmp/gemini_plugin_test_%s", ranstr)
	tmp := path
	ranstr = randomString(10)
	query_path := fmt.Sprintf("/tmp/gemini_plugin_test_%s", ranstr)
	arr := make([][]float32, 4000)
	for i := 0; i < len(arr); i++ {
		arr[i] = make([]float32, 96)
	}
	row_count, dim, aerr := Numpy_append_float32_array(path, arr, 96, 4000)
	if row_count != 4000 {
		log.Fatal("row mismatch")
	} else if dim != 96 {
		log.Fatal("dimension mismatch")
	} else if aerr != nil {
		log.Fatal(aerr)
	}
	row_count, dim, aerr = Numpy_append_float32_array(query_path, arr[:10], 96, 10)
	if row_count != 10 {
		log.Fatal("row mismatch")
	} else if dim != 96 {
		log.Fatal("dimension mismatch")
	} else if aerr != nil {
		log.Fatal(aerr)
	}
	search_type := "flat"
	// Import dataset tests
	t.Run("ImportDataset", func(t *testing.T) {
		// successful test
		tmp, err := Import_dataset(HOST, PORT, ALLOC, path, uint(bits), search_type, VERBOSE)
		dataset_id = tmp
		assert.Nilf(t, err, "Error importing dataset")
	})
	// Train status tests
	t.Run("TrainStatus", func(t *testing.T) {
		status, err := Train_status(HOST, PORT, ALLOC, dataset_id, VERBOSE)
		assert.Nilf(t, err, "Error getting train status")
		for status == "training" || status == "pending" {
			fmt.Println("currently", status, ": waiting...")
			time.Sleep(2 * time.Second)
			status, err = Train_status(HOST, PORT, ALLOC, dataset_id, VERBOSE)
			assert.Nilf(t, err, "Error getting train status while waiting on training")
			fmt.Println("\nstatus:", status)
		}
		assert.Equal(t, "completed", status, "Status should be \"completed\" after training")
	})
	// Load Dataset tests
	t.Run("LoadDataset", func(t *testing.T) {
		lstatus, err := Load_dataset(HOST, PORT, ALLOC, dataset_id, VERBOSE)
		assert.Nilf(t, err, "Error loading dataset")
		assert.Equal(t, "ok", lstatus, "Load status not \"ok\"")
	})
	// Import Query tests
	t.Run("ImportQueries", func(t *testing.T) {
		tmp, err := Import_queries(HOST, PORT, ALLOC, query_path, VERBOSE)
		query_id = tmp
		assert.Nilf(t, err, "Error importing queries")
	})
	// Focus Dataset
	t.Run("FocusDataset", func(t *testing.T) {
		err := Set_focus(HOST, PORT, ALLOC, dataset_id, VERBOSE)
		assert.Nilf(t, err, "Error setting dataset in focus")
	})
	// Search
	t.Run("Search", func(t *testing.T) {
		dists, inds, timing, err := Search(HOST, PORT, ALLOC, dataset_id, query_path, topk, VERBOSE)
		assert.Nilf(t, err, "Error querying dataset")
		assert.Equal(t, topk, uint(len(dists[0])), "Error in dimension mismatch with distances vector")
		assert.Equal(t, topk, uint(len(inds[0])), "Error in dimension mismatch with indices vector")
		assert.Less(t, timing, float32(0.01), "Search time suspiciously long")
	})
	// Unload Dataset
	t.Run("UnloadDataset", func(t *testing.T) {
		status, err := Unload_dataset(HOST, PORT, ALLOC, dataset_id, VERBOSE)
		assert.Nilf(t, err, "Error unloading dataset")
		assert.Equal(t, "ok", status, "Unload dataset status not \"ok\"")
	})
	// Delete Dataset
	t.Run("DeleteDataset", func(t *testing.T) {
		status, err := Delete_dataset(HOST, PORT, ALLOC, dataset_id, VERBOSE)
		assert.Nilf(t, err, "Error deleting dataset")
		assert.Equal(t, "ok", status, "Delete dataset status not \"ok\"")
	})
	// Delete Queries
	t.Run("DeleteQueries", func(t *testing.T) {
		status, err := Delete_queries(HOST, PORT, ALLOC, query_id, VERBOSE)
		assert.Nilf(t, err, "Error deleting queries")
		assert.Equal(t, "ok", status, "Delete query status not \"ok\"")
	})
}

// Testing import dataset with size less than 4k
// Error is expected during training

func TestFVSFunctions2(t *testing.T) {
	fmt.Println("\n\n----------TEST 2----------")
	ranstr := randomString(10)
	path = fmt.Sprintf("/tmp/gemini_plugin_test_%s", ranstr)
	arr := make([][]float32, 1000)
	for i := 0; i < len(arr); i++ {
		arr[i] = make([]float32, 96)
	}
	row_count, dim, aerr := Numpy_append_float32_array(path, arr, 96, 1000)
	if row_count != 1000 {
		log.Fatal("row mismatch")
	} else if dim != 96 {
		log.Fatal("dimension mismatch")
	} else if aerr != nil {
		log.Fatal(aerr)
	}
	search_type := "flat"

	// Import dataset test
	t.Run("ImportDataset", func(t *testing.T) {
		// successful test
		tmp, err := Import_dataset(HOST, PORT, ALLOC, path, uint(bits), search_type, VERBOSE)
		dataset_id = tmp
		assert.Nilf(t, err, "Error importing dataset")
	})
	// Train status tests
	t.Run("TrainStatus", func(t *testing.T) {
		status, err := Train_status(HOST, PORT, ALLOC, dataset_id, VERBOSE)
		assert.Nilf(t, err, "Error getting train status")
		for status == "training" || status == "pending" { // wait for training to finish, should end with "error"
			fmt.Println("currently", status, ": waiting...")
			time.Sleep(2 * time.Second)
			status, err = Train_status(HOST, PORT, ALLOC, dataset_id, VERBOSE)
			assert.Nilf(t, err, "Error getting train status while waiting on training")
		}
		// assert training ended with "error"
		assert.Equal(t, "error", status, "Train status should be \"error\" for a small dataset")
	})
	// Delete dataset, no test
	t.Run("DeleteDataset", func(t *testing.T) {
		status, err := Delete_dataset(HOST, PORT, ALLOC, dataset_id, VERBOSE)
		assert.Nilf(t, err, "Error deleting dataset")
		assert.Equal(t, "ok", status, "Delete dataset status should be \"ok\"")
	})
}

// Testing import dataset with odd bits
// Error is expected during dataset load
func TestFVSFunctions3(t *testing.T) {
	fmt.Println("\n\n----------TEST 3-----------")
	// setup for FVS testing
	bits := uint(137)
	path = tmp
	// import dataset
	t.Run("ImportDataset", func(t *testing.T) {
		tmp, err := Import_dataset(HOST, PORT, ALLOC, path, bits, search_type, VERBOSE)
		dataset_id = tmp
		assert.Nilf(t, err, "Error importing dataset")
	})
	// train status
	t.Run("TrainStatus", func(t *testing.T) {
		status, err := Train_status(HOST, PORT, ALLOC, dataset_id, VERBOSE)
		assert.Nilf(t, err, "Error getting training status")
		for status == "training" || status == "pending" {
			fmt.Println("currently", status, ": waiting...")
			time.Sleep(2 * time.Second)
			status, err = Train_status(HOST, PORT, ALLOC, dataset_id, VERBOSE)
			assert.Nilf(t, err, "Error getting train status while waiting on training")
		}
		assert.Equal(t, "completed", status, "Status should be \"completed\" after training")
	})
	// load dataset
	t.Run("LoadDataset", func(t *testing.T) {
		// status should be "error", err should be error "Server Internal Error"
		status, err := Load_dataset(HOST, PORT, ALLOC, dataset_id, VERBOSE)
		assert.NotNilf(t, err, "Error loading dataset")
		assert.Equal(t, "error", status, "Error status should be \"error\"")
	})
	// delete dataset
	t.Run("DeleteDataset", func(t *testing.T) {
		status, err := Delete_dataset(HOST, PORT, ALLOC, dataset_id, VERBOSE)
		assert.Nilf(t, err, "Error deleting dataset status should be \"ok\"")
		assert.Equal(t, "ok", status, "Delete dataset status should be \"ok\"")
	})
}

// test invalid search type (typo "fat")
func TestFVSFunctions4(t *testing.T) {
	fmt.Println("\n\n----------TEST 4-----------")
	path := "/mnt/nas1/fvs_benchmark_datasets/deep-10K.npy"
	search_type := "fat"

	// import dataset
	t.Run("ImportDataset", func(t *testing.T) {
		dataset_id, err := Import_dataset(HOST, PORT, ALLOC, path, uint(bits), search_type, VERBOSE)
		assert.NotNilf(t, err, "Error, search type is invalid, should not continue")
		assert.Equal(t, "", dataset_id, "Error, dataset_id should be nil")
	})
}
