package gemini

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"golang.org/x/exp/mmap"
)

// set constants and default variables for FVS
var HOST = "localhost"
var PORT = uint(7761)

// var ALLOC = "0b391a1a-b916-11ed-afcb-0242ac1c0002"
var ALLOC = "fd283b38-3e4a-11eb-a205-7085c2c5e516"
var VERBOSE = true

var topk = uint(5)
var bits = uint(128)
var search_type = "flat"
var path = "/mnt/nas1/fvs_benchmark_datasets/deep-10K.npy"
var query_path = "/mnt/nas1/fvs_benchmark_datasets/deep-queries-10.npy"
var dataset_id = ""
var query_id = ""

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

// Tests for FVS functions
// Test1: 10k dataset, 10 queries, 128 bits, flat search, topk 5
// no errors expected
func TestFVSFunctions1(t *testing.T) {
	// setup for FVS testing
	path := "/mnt/nas1/fvs_benchmark_datasets/deep-10K.npy"
	query_path := "/mnt/nas1/fvs_benchmark_datasets/deep-queries-10.npy"

	search_type := "flat"
	// Import dataset tests
	t.Run("ImportDataset", func(t *testing.T) {
		// successful test
		tmp, err := Import_dataset(HOST, PORT, ALLOC, path, bits, search_type, VERBOSE)
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
func TestFVSFunctions2(t *testing.T) {
	fmt.Println("\n\n----------TEST 2----------")
	time.Sleep(5 * time.Second)
	path := "/mnt/nas1/fvs_benchmark_datasets/deep-queries-1000.npy"
	search_type := "flat"

	// Import dataset test
	t.Run("ImportDataset", func(t *testing.T) {
		// successful test
		tmp, err := Import_dataset(HOST, PORT, ALLOC, path, bits, search_type, VERBOSE)
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
func TestFVSFunctions3(t *testing.T) {
	fmt.Println("\n\n----------TEST 3-----------")
	time.Sleep(5 * time.Second)
	// setup for FVS testing
	bits := uint(137)

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
	time.Sleep(5 * time.Second)
	path := "/mnt/nas1/fvs_benchmark_datasets/deep-10K.npy"
	search_type := "fat"

	// import dataset
	t.Run("ImportDataset", func(t *testing.T) {
		dataset_id, err := Import_dataset(HOST, PORT, ALLOC, path, bits, search_type, VERBOSE)
		assert.NotNilf(t, err, "Error, search type is invalid, should not continue")
		assert.Equal(t, "", dataset_id, "Error, dataset_id should be nil")
	})
}

//
