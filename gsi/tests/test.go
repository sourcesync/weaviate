package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	geminiplugin "github.com/gsi/weaviate/gemini_plugin"
	"golang.org/x/exp/mmap"
)

// Generate a random string useful for generate a temp filename that does not already exist
func randomString(length int) string {
	rand.Seed(time.Now().UnixNano())
	b := make([]byte, length+2)
	rand.Read(b)
	return fmt.Sprintf("%x", b)[2 : length+2]
}

func uint32_test() string {
	fmt.Println("Starting Numpy_append_uint32 testing")
	ranstr := randomString(10)
	ranfilepath := fmt.Sprintf("/tmp/gemini_plugin_test_%s", ranstr)
	_, ferr := os.Stat(ranfilepath)
	if ferr == nil {
		log.Fatal("ERROR: File exists")
	}
	fmt.Println("Got temp file path->", ranfilepath)
	arr := make([][]uint32, 1)
	arr[0] = make([]uint32, 96)
	fmt.Println(arr, arr[0])

	// build numpy file w/ array
	aerr := geminiplugin.Numpy_append_uint32_array(ranfilepath, arr, 96, 1)
	if aerr != nil {
		fmt.Println("Error creating numpy file with uint32 array", aerr)
	} else {
		fmt.Println("no data to return, success")
	}
	return ranfilepath
}

func numpy_read_float_test(ranfilepath string) {
	fmt.Println("Starting numpy READ FLOAT tests")
	readerAt, err := mmap.Open(ranfilepath)
	if err != nil {
		log.Fatal(err, "mmap readerAt error")
	}
	arr := make([][]float32, 1)
	arr[0] = make([]float32, 96)
	_, err = geminiplugin.Numpy_read_float32_array(readerAt, arr, int64(96), int64(0), int64(1), int64(128))
	if err != nil {
		log.Fatal(err, "function call error")
	}

}

func numpy_read_uint_test(ranfilepath string) {
	fmt.Println("Starting numpy READ UINT test")
	readerAt, err := mmap.Open(ranfilepath)
	if err != nil {
		log.Fatal(err, "mmap readerAt error")
	}
	arr := make([][]uint32, 1)
	arr[0] = make([]uint32, 96)
	_, err = geminiplugin.Numpy_read_uint32_array(readerAt, arr, int64(96), int64(0), int64(1), int64(128))
	if err != nil {
		log.Fatal(err, "function call error")
	}

}

func main() {

	// Create a temp file that will automatically delete itself when this app is done
	ranstr := randomString(10)
	ranfilepath := fmt.Sprintf("/tmp/gemini_plugin_test_%s", ranstr)
	_, ferr := os.Stat(ranfilepath)
	if ferr == nil {
		log.Fatal("ERROR: File exists.")
	}
	fmt.Println("Got temp file path->", ranfilepath)

	//
	// Create a float array of dims = 96
	//
	arr := make([][]float32, 1)
	arr[0] = make([]float32, 96)
	fmt.Println(arr, arr[0])

	//
	// Build a numpy file with the array
	//
	fmt.Println("Testing Numpy_append_float32_array")
	row_count, dim, aerr := geminiplugin.Numpy_append_float32_array(ranfilepath, arr, 96, 1)
	if aerr != nil {
		fmt.Println("Error creating numpy file with float32 array.", aerr)
	} else {
		fmt.Println("Got return data", row_count, dim)
	}

	fmt.Println("Done.")

	uintfp := uint32_test()
	numpy_read_uint_test(uintfp)
	numpy_read_float_test(ranfilepath)

}
