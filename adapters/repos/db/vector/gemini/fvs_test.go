package testing

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

func numpy_append_float_test() string {
	// create a temp file that will automatically delete itself when the app is done
	ranstr := randomString(10)
	ranfilepath := fmt.Sprintf("/tmp/gemini_plugin_test_%s", ranstr)
	_, ferr := os.Stat(ranfilepath)
	if ferr == nil {
		log.Fatal("ERROR: File exists.")
	}
	fmt.Println("Got temp file path->", ranfilepath)

	// create a float array of dims = 96
	arr := make([][]float32, 1)
	arr[0] = make([]float32, 96)
	fmt.Println(arr, arr[0])

	// build a numpy file with the array
	fmt.Println("Testing Numpy_append_float32_array")
	row_count, dim, aerr := geminiplugin.Numpy_append_float32_array(ranfilepath, arr, 96, 1)
	if aerr != nil {
		fmt.Println("Error creating numpy file with float32 array.", aerr)
	} else {
		fmt.Println("Got return data", row_count, dim)
	}

	fmt.Println("Done.")
	return ranfilepath
}

func numpy_append_uint_test() string {
	// create a temp file that will automatically delete itself when the app is done
	fmt.Println("Starting Numpy_append_uint32 testing")
	ranstr := randomString(10)
	ranfilepath := fmt.Sprintf("/tmp/gemini_plugin_test_%s", ranstr)
	_, ferr := os.Stat(ranfilepath)
	if ferr == nil {
		log.Fatal("ERROR: File exists")
	}
	fmt.Println("Got temp file path->", ranfilepath)

	// create a unint array of dims = 96
	arr := make([][]uint32, 1)
	arr[0] = make([]uint32, 96)
	fmt.Println(arr, arr[0])

	// build a numpy file with the array
	fmt.Println("Testing Numpy_append_uint32_array")
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
	// create mmap readerAt object for generated numpy file
	readerAt, err := mmap.Open(ranfilepath)
	if err != nil {
		log.Fatal(err, "mmap readerAt error")
	}
	// create a float array of dims = 96
	arr := make([][]float32, 1)
	arr[0] = make([]float32, 96)
	// read generated numpy file
	_, err = geminiplugin.Numpy_read_float32_array(readerAt, arr, int64(96), int64(0), int64(1), int64(128))
	if err != nil {
		log.Fatal(err, "function call error")
	}
}

func numpy_read_uint_test(ranfilepath string) {
	fmt.Println("Starting numpy READ UINT test")
	// create mmap readerAt object for generated numpy file
	readerAt, err := mmap.Open(ranfilepath)
	if err != nil {
		log.Fatal(err, "mmap readerAt error")
	}
	// create a uint array of dims = 96
	arr := make([][]uint32, 1)
	arr[0] = make([]uint32, 96)
	// read generated numpy file
	_, err = geminiplugin.Numpy_read_uint32_array(readerAt, arr, int64(96), int64(0), int64(1), int64(128))
	if err != nil {
		log.Fatal(err, "function call error")
	}
}

func main() {

	// numpy append/read function testing
	floatfp := numpy_append_float_test()
	uintfp := numpy_append_uint_test()
	numpy_read_float_test(floatfp)
	numpy_read_uint_test(uintfp)

	// FVS function testing
	host := "localhost"
	port := uint(7761)
	alloc := "0b391a1a-b916-11ed-afcb-0242ac1c0002"
	path := "/mnt/nas1/fvs_benchmark_datasets/deep-10K.npy"
	query_path := "/mnt/nas1/fvs_benchmark_datasets/deep-queries-10.npy"
	bits := uint(512)
	verbose := true
	fmt.Println("\nImporting dataset...")
	dataset_id, err := geminiplugin.Import_dataset(host, port, alloc, path, bits, verbose)
	if err != nil {
		log.Fatal(err, ", error with Import Dataset")
	}
	fmt.Println("\nTrain Status...")
	status, err := geminiplugin.Train_status(host, port, alloc, dataset_id, verbose)
	if err != nil {
		log.Fatal(err, ", error with train status")
	}
	for status == "training" {
		fmt.Println("still training, waiting 5 seconds...")
		time.Sleep(5 * time.Second)
		status, err = geminiplugin.Train_status(host, port, alloc, dataset_id, verbose)
		if err != nil {
			log.Fatal(err, ", error with train status")
		}
		fmt.Println("current status: ", status)
	}
	fmt.Println("\nLoading Dataset...")
	_, err = geminiplugin.Load_dataset(host, port, alloc, dataset_id, verbose)
	if err != nil {
		log.Fatal(err, ", error with Load Dataset")
	}
	fmt.Println("\nImporting queries...")
	qid, err := geminiplugin.Import_queries(host, port, alloc, query_path, verbose)
	if err != nil {
		log.Fatal(err, ", error with Import Queries")
	}
	fmt.Println("\nSetting focus to dataset ", dataset_id)
	err = geminiplugin.Set_focus(host, port, alloc, dataset_id, verbose)
	if err != nil {
		fmt.Println(err, ", error with focus dataset")
	}
	fmt.Println("\nSearching dataset")
	dists, inds, topk, err := geminiplugin.Search(host, port, alloc, dataset_id, query_path, uint(5), verbose)
	if err != nil {
		log.Fatal(err, " error with search")
	}
	fmt.Println("dists:", dists, " inds:", inds, " topk:", topk)
	fmt.Println("\nDeleting queries")
	ok, err := geminiplugin.Delete_queries(host, port, alloc, qid, verbose)
	if err != nil {
		log.Fatal(err, " error from Delete Queries")
	}
	fmt.Println(ok, "==ok")
	fmt.Println("Unloading dataset")
	status, err = geminiplugin.Unload_dataset(host, port, alloc, dataset_id, verbose)
	if err != nil {
		log.Fatal(err, " error with Unload dataset")
	}
	fmt.Println("status: ", status)
}
