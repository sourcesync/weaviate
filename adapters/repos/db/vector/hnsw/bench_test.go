package hnsw

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/kshedden/gonpy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/weaviate/weaviate/adapters/repos/db/vector/hnsw/distancer"
	ent "github.com/weaviate/weaviate/entities/vectorindex/hnsw"
)

const (
	datadir = "/mnt/nas1/fvs_benchmark_datasets"
	// csvpath = "/mnt/nas1/weaviate_benchmark_results/algo_direct/"
	csvpath = "/home/jacob/"
	multi   = true
	k       = 10
	dims    = 96
	gt_size = 100
)

func fileExists(fname string) bool {
	info, err := os.Stat(fname)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func WriteToCSV(data_name string, n uint, q uint, k int, ef int, loadTime float64, searchTime float64, avgRecall float32, t1 time.Time, t2 time.Time) {
	server, _ := os.Hostname()
	fname := fmt.Sprintf("%s%s_%s.csv", csvpath, server, data_name)
	if fileExists(fname) {
		fmt.Println("file exists")
	} else {
		fmt.Println("creating file", fname)
		file, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
		if err != nil {
			panic(err)
		}
		writer := csv.NewWriter(file)
		row := []string{"size", "query_count", "topK", "ef", "load_time", "search_time", "avg_recall", "server", "t1", "t2"}
		err = writer.Write(row)
		if err != nil {
			panic(err)
		}
		writer.Flush()
	}
	file, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
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
		fmt.Sprintf("%f", searchTime),
		fmt.Sprintf("%f", avgRecall),
		server,
		t1.Format("2006-01-02 15:04:05"),
		t2.Format("2006-01-02 15:04:05"),
	}
	fmt.Println(row)
	err = writer.Write(row)
	if err != nil {
		panic(err)
	}
	writer.Flush()
}

func ReadFloat32(path string, shape []uint) [][]float32 {
	r, err := gonpy.NewFileReader(path)
	if err != nil {
		panic(err)
	}
	var foo []float32
	foo, err = r.GetFloat32()
	var j, k int = 0, 0
	if err != nil {
		panic(err)
	}
	data := make([][]float32, shape[0])
	for i := range data {
		data[i] = make([]float32, shape[1])
	}
	for i := range foo {
		if i%int(shape[1]) == 0 && i != 0 {
			j += 1
			k = 0
		}
		data[j][k] = foo[i]
		k += 1
	}
	return data
}

func ReadInt32(path string, shape []uint) [][]int32 {
	r, err := gonpy.NewFileReader(path)
	if err != nil {
		panic(err)
	}
	var j, k int = 0, 0
	var foo []int32
	foo, err = r.GetInt32()
	if err != nil {
		panic(err)
	}
	data := make([][]int32, shape[0])
	for i := range data {
		data[i] = make([]int32, shape[1])
	}
	for i := range foo {
		if i%int(shape[1]) == 0 && i != 0 {
			j += 1
			k = 0
		}
		data[j][k] = foo[i]
		k += 1
	}
	return data
}

func run_queries(queryVectors [][]float32, gt [][]int32, index *hnsw, k int, ef int) float32 {
	var recalls []float32
	var ys, ns, total int = 0, 0, 0
	for i, vec := range queryVectors {
		res, _, err := index.knnSearchByVector(vec, int(k), ef, nil)
		if err != nil {
			panic(err)
		}
		var y, n int = 0, 0
		total += len(res)

		if i == 0 {
			fmt.Println("length of res:", len(res), " res[0]:", res[0])
		}
		for j := range res {
			if gt[i][j] == int32(res[j]) {
				y += 1
			} else {
				n += 1
			}
		}
		ys += y
		ns += n
		recalls = append(recalls, float32(y)/float32(len(res)))
	}
	var sum float32
	for _, recall := range recalls {
		sum += recall
	}
	avg_recall := sum / float32(len(recalls))
	fmt.Println("ys:", ys, "  ns:", ns, "  avg recall:", avg_recall)
	return avg_recall
}

var (
	data_size, _  = strconv.Atoi(os.Getenv("DATASIZE"))
	query_size, _ = strconv.Atoi(os.Getenv("QUERYSIZE"))
	start_size, _ = strconv.Atoi(os.Getenv("START"))
	increment, _  = strconv.Atoi(os.Getenv("INCREMENT"))
)

func TestBench(t *testing.T) {
	tmp := len(fmt.Sprintf("%d", data_size))
	var data_name string
	if tmp < 7 { // if data size is in the thousands name is size/1000 + K
		data_name = fmt.Sprintf("%dK", data_size/1000)
	} else if tmp < 10 { // if data size is in millions name is size/1000000 + M
		data_name = fmt.Sprintf("%dM", data_size/1000000)
	}

	fmt.Println("Benchmark Test:", data_name, data_size, query_size, "start time:", time.Now().Format("2006-01-02 15:04:05"))

	var data_size, query_size = uint(data_size), uint(query_size)
	data_path := fmt.Sprintf("%s/deep-%s.npy", datadir, data_name)
	query_path := fmt.Sprintf("%s/deep-queries-%d.npy", datadir, query_size)
	gt_path := fmt.Sprintf("%s/deep-%s-gt-%d.npy", datadir, data_name, query_size)
	paths := []string{data_path, query_path, gt_path}
	for _, path := range paths { // check files exist
		_, err := os.Stat(path)
		if err != nil {
			panic(err)
		}
	}

	fmt.Println("reading numpy files...")
	testVectors := ReadFloat32(data_path, []uint{data_size, dims})
	queryVectors := ReadFloat32(query_path, []uint{query_size, dims})
	gt := ReadInt32(gt_path, []uint{query_size, gt_size})

	// assertions for vector shape
	assert.Equal(t, int(data_size), len(testVectors))
	assert.Equal(t, int(query_size), len(queryVectors))

	// initialize hnsw index
	makeCL := MakeNoopCommitLogger
	vectorFunc := func(ctx context.Context, id uint64) ([]float32, error) {
		return testVectors[int(id)], nil
	}
	fmt.Println("creating hnsw index...")
	index, err := New(Config{
		RootPath:              "doesnt-matter-as-committlogger-is-mocked-out",
		ID:                    "unittest",
		MakeCommitLoggerThunk: makeCL,
		DistanceProvider:      distancer.NewCosineDistanceProvider(),
		VectorForIDThunk:      vectorFunc,
	}, ent.UserConfig{
		MaxConnections:   64,
		EFConstruction:   128,
		DynamicEFMin:     100,
		DynamicEFMax:     500,
		DynamicEFFactor:  8,
		FlatSearchCutoff: 40000,
		Distance:         "cos",
	})
	require.Nil(t, err)
	ef := index.autoEfFromK(int(k))

	t1 := time.Now()
	for i, vec := range testVectors[:start_size] {
		err := index.Add(uint64(i), vec)
		require.Nil(t, err)
	}

	load_time := time.Since(t1).Seconds()
	t2 := time.Now()
	fmt.Println("running queries...")
	avg_recall := run_queries(queryVectors, gt, index, k, ef)
	search_time := time.Since(t2).Seconds()
	fmt.Println("search time:", search_time, " seconds")
	WriteToCSV(data_name, data_size, query_size, k, ef, load_time, search_time, avg_recall, t1, t2)

	// loop for queries, break after 1 iteration if "multi" is false
	for i := 0; i < 4; i++ {
		if !multi {
			break
		}
		// read groundtruth from current data size
		gt_path := fmt.Sprintf("%s/deep-%s-gt-%d.npy", datadir, data_name, query_size)
		gt := ReadInt32(gt_path, []uint{query_size, gt_size})
		assert.Equal(t, int(query_size), len(gt))

		fmt.Println("loading vectors...")
		t1 := time.Now()
		for j, vec := range testVectors[start_size+increment*i : start_size+increment*(i+1)] {
			err := index.Add(uint64(j), vec)
			require.Nil(t, err)
		}
		fmt.Println(i)

		load_time = time.Since(t1).Seconds()
		fmt.Println("load time:", load_time, " seconds")

		t2 := time.Now()
		fmt.Println("running queries...")
		avg_recall := run_queries(queryVectors, gt, index, k, ef)
		search_time := time.Since(t2).Seconds()
		fmt.Println("search time:", search_time, " seconds")
		WriteToCSV(data_name, data_size, query_size, k, ef, load_time, search_time, avg_recall, t1, t2)
	}
	fmt.Println("Done.")
}
