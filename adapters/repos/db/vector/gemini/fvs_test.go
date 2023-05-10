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
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"testing"
	"time"

	"github.com/gorilla/mux"
	"github.com/stretchr/testify/assert"
	"golang.org/x/exp/mmap"
)

const (
	HOST = "localhost"
	PORT = uint(7760)
	// ALLOC = "fd283b38-3e4a-11eb-a205-7085c2c5e516"
	ALLOC   = "0b391a1a-b916-11ed-afcb-0242ac1c0002"
	VERBOSE = true
	FAKE    = true
)

var (
	topk         = uint(5)
	bits         = float64(128)
	search_type  = "flat"
	dataset_path string
	path         string
	dataset_id   string
	query_id     string
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
	router.HandleFunc("/v1.0/dataset/list", handleListDatasets)
	router.HandleFunc("/v1.0/board/allocation/list", handleListLoaded)

	ctx := context.Background()
	graceperiod := 1 * time.Second
	httpAddr := ":7760"

	srv := &http.Server{
		Addr:    httpAddr,
		Handler: router,
	}

	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGINT, syscall.SIGTERM)

	if VERBOSE {
		fmt.Println("starting http server")
	}
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()

	t.Parallel()
	ctxTimeout, cancel := context.WithTimeout(ctx, graceperiod)
	defer func() {
		cancel()
		if VERBOSE {
			fmt.Println("STOPPING SERVER")
		}
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
	if VERBOSE {
		fmt.Println("----------TEST 1----------")
	}
	ranstr := randomString(10)
	path = fmt.Sprintf("/home/public/gemini_plugin_test_%s.npy", ranstr)
	dataset_path = path
	ranstr = randomString(10)
	query_path := fmt.Sprintf("/home/public/gemini_plugin_test_%s.npy", ranstr)
	arr := make([][]float32, 4001)
	for i := 0; i < len(arr); i++ {
		arr[i] = make([]float32, 96)
	}
	row_count, dim, aerr := Numpy_append_float32_array(path, arr, 96, 4001)
	if aerr != nil {
		log.Fatal(aerr)
	} else if row_count != 4001 {
		log.Fatal("row mismatch")
	} else if dim != 96 {
		log.Fatal("dimension mismatch")
	}
	row_count, dim, aerr = Numpy_append_float32_array(query_path, arr[:10], 96, 10)
	if aerr != nil {
		log.Fatal(aerr)
	} else if row_count != 10 {
		log.Fatal("row mismatch")
	} else if dim != 96 {
		log.Fatal("dimension mismatch")
	}
	search_type := "flat"
	defer os.Remove(query_path)

	// List datasets test
	t.Run("ListDatasets", func(t *testing.T) {
		count, dsets, err := List_datasets(HOST, PORT, ALLOC, VERBOSE)
		assert.Nilf(t, err, "Error listing datasets")
		assert.GreaterOrEqual(t, count, 0, "Dataset count should be 0 or positive int")
		if count > 0 {
			for _, v := range dsets {
				assert.Contains(t, v, "id")
			}
		}
	})
	// Unload all datasets test
	t.Run("UnloadDatasets", func(t *testing.T) {
		status, err := Unload_loaded(HOST, PORT, ALLOC, true, VERBOSE)
		assert.Nilf(t, err, "Error unloading datasets")
		assert.Equal(t, "ok", status, "Error unloading datasets")
	})
	// Import dataset tests
	t.Run("ImportDataset", func(t *testing.T) {
		tmp, err := Import_dataset(HOST, PORT, ALLOC, path, uint(bits), search_type, VERBOSE)
		dataset_id = tmp
		assert.Nilf(t, err, "Error importing dataset")
		assert.GreaterOrEqual(t, len(dataset_id), 30, "Error importing dataset")
	})
	// Train status tests
	t.Run("TrainStatus", func(t *testing.T) {
		status, err := Train_status(HOST, PORT, ALLOC, dataset_id, VERBOSE)
		assert.Nilf(t, err, "Error getting train status")
		for status == "training" || status == "pending" || status == "loading" {
			if VERBOSE {
				fmt.Println("currently", status, ": waiting...")
			}
			time.Sleep(3 * time.Second)
			status, err = Train_status(HOST, PORT, ALLOC, dataset_id, VERBOSE)
			assert.Nilf(t, err, "Error getting train status while waiting on training")
			if VERBOSE {
				fmt.Println("\nstatus:", status)
			}
		}
		assert.Equal(t, "completed", status, "Status should be \"completed\" after training")
	})
	// Load Dataset tests
	t.Run("LoadDataset", func(t *testing.T) {
		lstatus, err := Load_dataset(HOST, PORT, ALLOC, dataset_id, VERBOSE)
		assert.Nilf(t, err, "Error loading dataset")
		assert.Equal(t, "ok", lstatus, "Load status not \"ok\"")
	})
	// List Loaded datasets tests
	t.Run("ListLoaded", func(t *testing.T) {
		count, loaded, err := List_loaded(HOST, PORT, ALLOC, VERBOSE)
		assert.Nilf(t, err, "Error listing loaded datasets")
		assert.Equal(t, 1, count, "Error: should be 1 loaded dataset")
		assert.Contains(t, loaded, "loadedDatasets", "Error: should have \"loadedDatasets\" key in response")
		dsets := loaded["loadedDatasets"].([]interface{})[0].(map[string]interface{})
		assert.Equal(t, dsets["datasetId"], dataset_id)
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
		assert.Less(t, timing, float32(1.0), "Search time suspiciously long")
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
	if VERBOSE {
		fmt.Println("\n\n----------TEST 2----------")
	}
	ranstr := randomString(10)
	path = fmt.Sprintf("/home/public/gemini_plugin_test_%s.npy", ranstr)
	defer os.Remove(path)
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
	if _, err := os.Stat(path); err != nil {
		fmt.Println("could not find path=", path)
	}

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
			if VERBOSE {
				fmt.Println("currently", status, ": waiting...")
			}
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
	if VERBOSE {
		fmt.Println("\n\n----------TEST 3-----------")
	}
	// setup for FVS testing
	bits := uint(137)
	path := dataset_path
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
			if VERBOSE {
				fmt.Println("currently", status, ": waiting...")
			}
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
	if VERBOSE {
		fmt.Println("\n\n----------TEST 4-----------")
	}
	search_type := "fat"
	path := dataset_path
	// import dataset
	t.Run("ImportDataset", func(t *testing.T) {
		dataset_id, err := Import_dataset(HOST, PORT, ALLOC, path, uint(bits), search_type, VERBOSE)
		assert.NotNilf(t, err, "Error, search type is invalid, should not continue")
		assert.Equal(t, "", dataset_id, "Error, dataset_id should be nil")
	})
	err := os.Remove(path)
	if err != nil {
		log.Fatal(err)
	}
}
