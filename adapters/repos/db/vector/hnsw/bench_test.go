package hnsw

import (
	"context"
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/weaviate/weaviate/adapters/repos/db/vector/hnsw/distancer"
	ent "github.com/weaviate/weaviate/entities/vectorindex/hnsw"
	"golang.org/x/exp/mmap"
)

const (
	datadir = "/mnt/nas1/fvs_benchmark_datasets" // CHANGE for new data
	// csvpath = "/mnt/nas1/weaviate_benchmark_results/algo_direct/"
	k       = 10
	dims    = 96
	gt_size = 100
)

var (
	csvpath       = os.Getenv("CSVPATH")
	data_size, _  = strconv.Atoi(os.Getenv("DATASIZE"))
	query_size, _ = strconv.Atoi(os.Getenv("QUERYSIZE"))
	start_size, _ = strconv.Atoi(os.Getenv("START"))
	increment, _  = strconv.Atoi(os.Getenv("INCREMENT"))
	multi, _      = strconv.ParseBool(os.Getenv("MULTI"))
)

func fileExists(fname string) bool {
	info, err := os.Stat(fname)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func WriteToCSV(csvpath string, dset string, num_recs int, ef int, q int, inds []uint64, searchTime float64, ts time.Time, trainVectors float64, wall float64) {
	server, _ := os.Hostname()
	fname := fmt.Sprintf("%salgodirect_hnsw_%s_%s_%d_%d_%f_%f.csv", csvpath, server, dset, num_recs, ef, trainVectors, wall)
	if !fileExists(fname) {
		fmt.Println("creating file", fname)
		file, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
		if err != nil {
			panic(err)
		}
		writer := csv.NewWriter(file)
		row := []string{"ts", "q_index", "search_time", "inds"}
		err = writer.Write(row)
		if err != nil {
			panic(err)
		}
		writer.Flush()
	}
	file, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		panic(err)
	}

	writer := csv.NewWriter(file)

	inds_str := fmt.Sprintf(
		"%d-%d-%d-%d-%d-%d-%d-%d-%d-%d",
		inds[0], inds[1], inds[2], inds[3], inds[4], inds[5], inds[6], inds[7], inds[8], inds[9],
	)

	row := []string{
		fmt.Sprintf("%d", time.Now().Unix()),
		fmt.Sprintf("%d", q),
		fmt.Sprintf("%f", searchTime),
		inds_str,
	}
	err = writer.Write(row)
	if err != nil {
		panic(err)
	}
	writer.Flush()
}

// Read a float32 array from data stored in numpy format
func Numpy_read_float32_array(f *mmap.ReaderAt, arr [][]float32, dim int64, index int64, count int64, offset int64) (int64, error) {
	// Iterate rows
	for j := 0; j < int(count); j++ {
		// Read consecutive 4 byte array into uint32 array, up to dims
		for i := 0; i < len(arr[j]); i++ {

			// Declare 4-byte array
			bt := []byte{0, 0, 0, 0}

			// Compute offset for next uint32
			r_offset := offset + (int64(j)+index)*dim*4 + int64(i)*4

			_, err := f.ReadAt(bt, r_offset)
			if err != nil {
				return 0, fmt.Errorf("error reading file at offset:%v, %v", r_offset, err)
			}

			bits := binary.LittleEndian.Uint32(bt)
			arr[j][i] = math.Float32frombits(bits)
		}
	}

	return dim, nil
}

func name_dataset(data_size int) string {
	tmp := len(fmt.Sprintf("%d", data_size))
	var data_name string
	if tmp < 7 { // if data size is in the thousands name is size/1000 + K
		data_name = fmt.Sprintf("%dK", data_size/1000)
	} else if tmp < 10 { // if data size is in millions name is size/1000000 + M
		data_name = fmt.Sprintf("%dM", data_size/1000000)
	} else {
		data_name = fmt.Sprintf("%dB", data_size/1000000000)
	}
	return data_name
}

func TestBench(t *testing.T) {

	data_name := name_dataset(data_size)
	// create data readers
	data_path := fmt.Sprintf("%s/deep-%s.npy", datadir, data_name) // CHANGE for new data
	fmt.Println(data_path)
	data_reader, ferr := mmap.Open(data_path)
	assert.Nil(t, ferr)
	query_path := fmt.Sprintf("%s/deep-queries-%d.npy", datadir, query_size) // CHANGE for new data
	fmt.Println(query_path)
	query_reader, ferr := mmap.Open(query_path)
	assert.Nil(t, ferr)

	// check files exist
	paths := []string{data_path, query_path}
	for _, path := range paths {
		_, err := os.Stat(path)
		assert.Nil(t, err)
	}

	// initialize empty arrays
	trainVectors := make([][]float32, start_size)
	for i := range trainVectors {
		trainVectors[i] = make([]float32, dims)
	}
	queryVectors := make([][]float32, query_size)
	for i := range queryVectors {
		queryVectors[i] = make([]float32, dims)
	}

	// read data to arrays
	fmt.Println("reading numpy files...", time.Now().Format("15:04:05"))
	_, err := Numpy_read_float32_array(data_reader, trainVectors, int64(dims), int64(0), int64(start_size), int64(128))
	assert.Nil(t, err)
	_, err = Numpy_read_float32_array(query_reader, queryVectors, int64(dims), int64(0), int64(query_size), int64(128))
	assert.Nil(t, err)

	// initialize hnsw index
	makeCL := MakeNoopCommitLogger
	vectorFunc := func(ctx context.Context, id uint64) ([]float32, error) {
		return trainVectors[id], nil
	}
	index, err := New(Config{
		RootPath:              "doesnt-matter-as-committlogger-is-mocked-out",
		ID:                    "unittest",
		MakeCommitLoggerThunk: makeCL,
		DistanceProvider:      distancer.NewCosineDistanceProvider(),
		VectorForIDThunk:      vectorFunc,
	}, ent.UserConfig{
		EF:                    -1,
		MaxConnections:        64,
		EFConstruction:        64,
		DynamicEFMin:          100,
		DynamicEFMax:          500,
		DynamicEFFactor:       8,
		FlatSearchCutoff:      40000,
		VectorCacheMaxObjects: 0,
		Distance:              "cosine",
	})
	require.Nil(t, err)
	ef_array := []int{64, 128, 256, 512}

	// assertions for vector shape
	assert.Equal(t, int(start_size), len(trainVectors))
	assert.Equal(t, int(query_size), len(queryVectors))

	fmt.Println("Benchmark Test:", data_name, data_size, query_size, "start time:", time.Now().Format("2006-01-02 15:04:05"))
	var size, curr int = start_size, 0

	// loop for queries, break after 1 iteration if "multi" is false
	batch_size := 10000
	for size <= data_size {
		fmt.Println("loading vectors", curr, ":", size, "to hnsw index...")
		t1 := time.Now()
		var load_time time.Duration // initiali
		for i := 0; i < len(trainVectors); i += batch_size {
			fmt.Println("adding vecs:", i, ":", i+batch_size)
			t2, err := index.AddBatch(uint64(i), trainVectors[i:i+batch_size])
			load_time += t2 // append load time
			require.Nil(t, err)
		}

		wall_time := time.Since(t1)
		fmt.Println("running queries...")
		for _, ef := range ef_array {
			for i := 0; i < 1000; i++ { // loop over queries
				t1 = time.Now()
				inds, _, err := index.knnSearchByVector(queryVectors[i], k, ef, nil)
				searchTime := time.Since(t1).Seconds()
				if err != nil {
					panic(err)
				}
				WriteToCSV(csvpath, data_name, size, ef, i, inds, searchTime, time.Now(), float64(load_time.Seconds()), float64(wall_time.Seconds()))
			}
		}

		// if multi-run benchmark, increment size and continue
		if !multi {
			break
		}
		if curr == 0 {
			curr += size
		} else {
			curr += increment
		}
		size += increment
		if size > data_size {
			break
		}

		// append new vectors to trainVectors
		tmp := make([][]float32, increment)
		for i := range tmp {
			tmp[i] = make([]float32, dims)
		}

		_, err = Numpy_read_float32_array(data_reader, tmp, dims, int64(curr), int64(size-curr), int64(8))
		assert.Nil(t, err)
		trainVectors = append(trainVectors, tmp...)
		fmt.Println("data length: ", len(trainVectors))

	}
	fmt.Println("Done.")
}
