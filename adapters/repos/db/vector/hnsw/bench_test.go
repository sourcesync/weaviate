package hnsw

import (
	"context"
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"sync"
	"testing"
	"time"

	mmapgo "github.com/edsrzf/mmap-go"
	"github.com/pkg/errors"
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
	cpus, _       = strconv.Atoi(os.Getenv("CPUS"))
)

func fileExists(fname string) bool {
	info, err := os.Stat(fname)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func WriteIndsNpy(size int, inds [][]uint64, i int) {
	server, _ := os.Hostname()
	data_name := name_dataset(size)
	fname := fmt.Sprintf("%s%s_%s_indices_%d.npy", csvpath, server, data_name, i)
	arr := make([][]uint32, len(inds))
	for i := range inds {
		arr[i] = make([]uint32, len(inds[i]))
		for j := range inds[i] {
			arr[i][j] = uint32(inds[i][j])
		}
	}
	err := Numpy_append_uint32_array(fname, arr, k, int64(len(arr)))
	if err != nil {
		panic(err)
	}
}

func WriteToCSV(data_name string, n int, q int, k int, ef int, loadTime float64, wallTime float64, searchTime float64) {
	server, _ := os.Hostname()
	fname := fmt.Sprintf("%s%s_algodirect.csv", csvpath, server)
	if !fileExists(fname) {
		fmt.Println("creating file", fname)
		file, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
		if err != nil {
			panic(err)
		}
		writer := csv.NewWriter(file)
		row := []string{"size", "query_count", "topK", "ef", "load_time", "wall_time", "search_time", "server", "stamp"}
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

	row := []string{
		fmt.Sprintf("%d", n),
		fmt.Sprintf("%d", q),
		fmt.Sprintf("%d", k),
		fmt.Sprintf("%d", ef),
		fmt.Sprintf("%f", loadTime),
		fmt.Sprintf("%f", wallTime),
		fmt.Sprintf("%f", searchTime),
		server,
		time.Now().Format("2006-01-02 15:04:05"),
	}
	err = writer.Write(row)
	if err != nil {
		panic(err)
	}
	writer.Flush()
}

// Write a uint32 array to a file in numpy format
func Numpy_append_uint32_array(fname string, arr [][]uint32, dim int64, count int64) error {
	preheader := []byte{0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00, 0x76, 0x00}
	fmt_header := "{'descr': '<i4', 'fortran_order': False, 'shape': (%d, %d), }"
	empty := []byte{0x20}
	fin := []byte{0x0a}

	// Check if file exists
	fexists := true
	_, err := os.Stat(fname)
	if os.IsNotExist(err) {
		fexists = false
	}

	// Get file descriptor
	var f *os.File = nil
	if fexists {
		// Open file
		f, err = os.OpenFile(fname, os.O_RDWR, 0o755)
		if err != nil {
			return fmt.Errorf("error openingfile: %v", err)
		}
	} else {
		// Create file
		f, err = os.Create(fname)
		if err != nil {
			return errors.Wrap(err, "error creating file in Numpy_append_uint32_array")
		}
		// Create header area
		err = f.Truncate(int64(128))
		if err != nil {
			return errors.Wrap(err, "error resizing file for header in Numpy_append_uint32_array")
		}
	}
	defer f.Close()

	// Get file size
	fi, err := f.Stat()
	if err != nil {
		return errors.Wrap(err, "error get file stats in Numpy_append_uint32_array.")
	}
	file_size := int64(fi.Size())

	// Get row count
	data_size := file_size - 128
	row_count := data_size / (dim * 4)
	new_row_count := row_count + count

	// Resize file
	new_size := file_size + dim*4*count
	err = f.Truncate(int64(new_size))
	if err != nil {
		return fmt.Errorf("error resizing file in Numpy_append_uint32_array: %v", err)
	}

	// Memory map the new file
	mem, err := mmapgo.Map(f, mmapgo.RDWR, 0)
	if err != nil {
		return errors.Wrap(err, "error mmapgo.Map in Numpy_append_uint32_array")
	}
	defer mem.Unmap()

	// Create the new header
	header := fmt.Sprintf(fmt_header, new_row_count, dim)

	// Write the numpy header info
	idx := 0
	for i := 0; i < len(preheader); i++ {
		mem[idx] = preheader[i]
		idx += 1
	}
	for i := 0; i < len(header); i++ {
		mem[idx] = header[i]
		idx += 1
	}
	for i := idx; i < 128; i++ {
		mem[idx] = empty[0]
		idx += 1
	}
	mem[127] = fin[0]

	// append the arrays
	idx = int(128 + data_size)
	for j := 0; j < int(count); j++ {
		for i := 0; i < len(arr[j]); i++ {
			bt := []byte{0, 0, 0, 0}
			binary.LittleEndian.PutUint32(bt, arr[j][i])
			mem[idx] = bt[0]
			mem[idx+1] = bt[1]
			mem[idx+2] = bt[2]
			mem[idx+3] = bt[3]
			idx += 4
		}
	}

	mem.Flush()

	return nil
}

// Read a uint32 array from data stored in numpy format
func Numpy_read_uint32_array(f *mmap.ReaderAt, arr [][]uint32, dim int64, index int64, count int64, offset int64) (int64, error) {
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
				return 0, errors.Wrapf(err, "error reading file at offset: %d, %v", r_offset, err)
			}

			arr[j][i] = binary.LittleEndian.Uint32(bt)
		}
	}

	return dim, nil
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

func speed_test(index *hnsw, k int, ef int, cpus int, queries [][]float32) Results {
	var times []time.Duration
	m := &sync.Mutex{}
	queues := make([][][]float32, cpus)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < len(queries); i++ {
		query := queries[i]
		worker := i % cpus
		queues[worker] = append(queues[worker], query)
	}
	wg := &sync.WaitGroup{}
	before := time.Now()
	for _, queue := range queues {
		wg.Add(1)
		go func(queue [][]float32) {
			defer wg.Done()

			for _, query := range queue {
				before := time.Now()
				_, _, err := index.knnSearchByVector(query, k, ef, nil)
				if err != nil {
					panic(err)
				}
				took := time.Since(before)
				m.Lock()
				times = append(times, took)
				m.Unlock()
			}
		}(queue)
	}

	wg.Wait()

	return analyze(times, time.Since(before))
}

var targetPercentiles = []int{50, 90, 95, 98, 99}

type Results struct {
	Min               time.Duration
	Max               time.Duration
	Mean              time.Duration
	Took              time.Duration
	QueriesPerSecond  float64
	Percentiles       []time.Duration
	PercentilesLabels []int
	Total             int
	Successful        int
	Failed            int
	Parallelization   int
}

func analyze(times []time.Duration, total time.Duration) Results {
	out := Results{Min: math.MaxInt64, PercentilesLabels: targetPercentiles}
	var sum time.Duration

	for _, time := range times {
		if time < out.Min {
			out.Min = time
		}
		if time > out.Max {
			out.Max = time
		}
		out.Successful++
		sum += time
	}

	out.Total = query_size
	out.Failed = query_size - out.Successful
	out.Parallelization = cpus
	out.Mean = sum / time.Duration(len(times))
	out.Took = total
	out.QueriesPerSecond = float64(len(times)) / float64(float64(total)/float64(time.Second))

	sort.Slice(times, func(a, b int) bool {
		return times[a] < times[b]
	})

	percentilePos := func(percentile int) int {
		return int(float64(len(times)*percentile)/100) + 1
	}

	out.Percentiles = make([]time.Duration, len(targetPercentiles))
	for i, percentile := range targetPercentiles {
		pos := percentilePos(percentile)
		if pos >= len(times) {
			pos = len(times) - 1
		}
		out.Percentiles[i] = times[pos]
	}

	return out
}

func run_queries(queryVectors [][]float32, index *hnsw, k int, ef int) [][]uint64 {
	arr := make([][]uint64, len(queryVectors)) // initialize array for returned indices

	for i, vec := range queryVectors {
		inds, _, err := index.knnSearchByVector(vec, int(k), ef, nil)
		arr[i] = inds
		if err != nil {
			panic(err)
		}
	}
	return arr
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
	testVectors := make([][]float32, start_size)
	for i := range testVectors {
		testVectors[i] = make([]float32, dims)
	}
	queryVectors := make([][]float32, query_size)
	for i := range queryVectors {
		queryVectors[i] = make([]float32, dims)
	}

	fmt.Println("reading numpy files...", time.Now().Format("15:04:05"))
	_, err := Numpy_read_float32_array(data_reader, testVectors, int64(dims), int64(0), int64(start_size), int64(128))
	assert.Nil(t, err)
	_, err = Numpy_read_float32_array(query_reader, queryVectors, int64(dims), int64(0), int64(query_size), int64(128))
	assert.Nil(t, err)

	// initialize hnsw index
	makeCL := MakeNoopCommitLogger
	vectorFunc := func(ctx context.Context, id uint64) ([]float32, error) {
		return testVectors[id], nil
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
	assert.Equal(t, int(start_size), len(testVectors))
	assert.Equal(t, int(query_size), len(queryVectors))

	fmt.Println("Benchmark Test:", data_name, data_size, query_size, "start time:", time.Now().Format("2006-01-02 15:04:05"))
	var size, curr int = start_size, 0

	// loop for queries, break after 1 iteration if "multi" is false
	batch_size := 10000
	for size <= data_size {
		fmt.Println("loading vectors", curr, ":", size, "to hnsw index...")
		t1 := time.Now()
		var load_time time.Duration
		for i := 0; i < len(testVectors); i += batch_size {
			fmt.Println("adding vecs:", i, ":", i+batch_size)
			t2, err := index.AddBatch(uint64(i), testVectors[i:i+batch_size])
			load_time += t2
			require.Nil(t, err)
		}

		wall_time := time.Since(t1)
		fmt.Println("running queries...")
		for _, ef := range ef_array {
			t2 := time.Now()
			inds := run_queries(queryVectors, index, k, ef)
			search_time := time.Since(t2).Seconds()
			fmt.Println("search time:", search_time, " seconds")
			WriteToCSV(data_name, size, query_size, k, ef, load_time.Seconds(), wall_time.Seconds(), search_time)
			WriteIndsNpy(size, inds, ef)
		}

		fmt.Println("Starting speed test...")
		for _, ef := range ef_array {
			results := speed_test(index, k, ef, cpus, queryVectors)
			fmt.Println(results)
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

		// append new vectors to testVectors
		tmp := make([][]float32, increment)
		for i := range tmp {
			tmp[i] = make([]float32, dims)
		}

		_, err = Numpy_read_float32_array(data_reader, tmp, dims, int64(curr), int64(size-curr), int64(8))
		assert.Nil(t, err)
		testVectors = append(testVectors, tmp...)
		fmt.Println("data length: ", len(testVectors))

	}
	fmt.Println("Done.")
}
