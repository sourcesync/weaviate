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

	// Run all unit tests in parallel for speed
	t.Parallel()

	// Run a unit test for the function "Numpy_append_float32_array"
	t.Run("NumpyAppendFloat32", func(t *testing.T) {

		//
		// Prepare for the test
		//

		// get a temp file name
		ranstr := randomString(10)
		ranfilepath := fmt.Sprintf("/tmp/gemini_plugin_test_%s", ranstr)

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

		// Start float32 read test
		readerAt, ferr := mmap.Open(ranfilepath)

		// expecting nil for ferr
		assert.Nilf(t, ferr, "Got error for opening numpy file")

		// read numpy file and store dims
		rdim, aerr := Numpy_read_float32_array(readerAt, arr, int64(96), int64(0), int64(1), int64(128))

		// nil for aerr
		assert.Nilf(t, aerr, "Got error for Numpy_read_float32_array")

		// 96 for dim
		assert.Equal(t, rdim, int64(96), "Expecting dims of 96")

		// if we get here, the unit test has passed all checks!
		//
		// Unit test cleanup
		//
		derr := os.Remove(ranfilepath)
		assert.Nilf(t, derr, "Could not delete the temp file")

	})

	// Run a unit test for the function "Numpy_append_int32_array"
	t.Run("NumpyAppendFloat32", func(t *testing.T) {

		ranstr := randomString(10)
		ranfilepath := fmt.Sprintf("/tmp/gemini_plugin_test_%s", ranstr)
		_, ferr := os.Stat(ranfilepath)
		assert.NotNilf(t, ferr, "The file alerady exists")
		// create uint array
		arr := make([][]uint32, 1)
		arr[0] = make([]uint32, 96)
		aerr := Numpy_append_uint32_array(ranfilepath, arr, 96, 1)
		assert.Nilf(t, aerr, "Got error for Numpy_append_uint32_array")
		derr := os.Remove(ranfilepath)
		assert.Nilf(t, derr, "Could not delete the temp file")
	})

}
