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
		assert.Equal(t, row_count, 1, "Expecting row_count of 1")

		// we are expecting a specific "dim" value
		assert.Equal(t, dim, 96, "Expecting dim of 96")
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
		assert.Equal(t, rdim, int64(96), "Expecting dims of 96")

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
		assert.Equal(t, dim, int64(96), "Expecting dims of 96")

		// all tests passed, cleanup
		derr := os.Remove(ranfilepath)
		assert.Nilf(t, derr, "Could not delete the temp file")
	})
	// setup for FVS function testing
	host := "localhost"
	port := uint(7761)
	// alloc := "0b391a1a-b916-11ed-afcb-0242ac1c0002"
	alloc := "fd283b38-3e4a-11eb-a205-7085c2c5e516"
	path := "/mnt/nas1/fvs_benchmark_datasets/deep-10K.npy"
	query_path := "/mnt/nas1/fvs_benchmark_datasets/deep-queries-10.npy"
	bits := uint(128)
	verbose := true
	search_type := "flat"
	topk := uint(5)
	dataset_id := ""

	// Unit tests for FVS functions
	// Import Dataset tests
	t.Run("ImportDataset", func(t *testing.T) {

		// import dataset
		dataset_id, err := Import_dataset(host, port, alloc, path, bits, search_type, verbose)
		assert.Nilf(t, err, "Error importing dataset")

	})

	// Train status tests
	t.run("TrainStatus", func(t *testing.T) {
		status, err := Train_status(host, port, alloc, dataset_id, verbose)
		assert.Nilf(t, err, "Error getting train status")
		// wait for training to finish
		for status == "training" {
			time.Sleep(2 * time.Second)
			status, err = Train_status(host, port, alloc, dataset_id, verbose)
			assert.Nilf(t, err, "Error getting train status while waiting for training to finish")
		}
	})
		

		// Load dataset
		lstatus, err := Load_dataset(host, port, alloc, dataset_id, verbose)
		assert.Nilf(t, err, "error loading dataset")
		assert.Equal(t, lstatus, "ok", "error with load status")

		// Import queries
		query_id, err := Import_queries(host, port, alloc, query_path, verbose)
		assert.Nilf(t, err, "Error importing queries")

		// Focus dataset
		err = Set_focus(host, port, alloc, dataset_id, verbose)
		assert.Nilf(t, err, "Error setting dataset focus")

		// Searching dataset
		dists, inds, timing, err := Search(host, port, alloc, dataset_id, query_path, topk, verbose)
		assert.Nilf(t, err, "Error with Search")
		assert.Equal(t, len(dists[0]), 5, "Dimension mismatch with distances array")
		assert.Equal(t, len(inds[0]), 5, "Dimension mismatch with indices array")
		assert.Less(t, timing, float32(0.01), "Search time suspiciously long")

		// Unload dataset
		status, err = Unload_dataset(host, port, alloc, dataset_id, verbose)
		assert.Nilf(t, err, "Error unloading dataset")
		assert.Equal(t, status, "ok", "Unload dataset status not \"ok\"")

		// Delete dataset
		status, err = Delete_dataset(host, port, alloc, dataset_id, verbose)
		assert.Nilf(t, err, "Error deleting dataset")
		assert.Equal(t, status, "ok", "Delete dataset status not \"ok\"")

		// Delete queries
		status, err = Delete_queries(host, port, alloc, query_id, verbose)
		assert.Nilf(t, err, "Error deleteing queries")
		assert.Equal(t, status, "ok", "Delete query status not \"ok\"")
	})
}
