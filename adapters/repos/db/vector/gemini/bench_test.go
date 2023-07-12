package gemini

import (
	"encoding/csv"
	"fmt"
    "strings"
    "strconv"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	ent "github.com/weaviate/weaviate/entities/vectorindex/gemini"
	"golang.org/x/exp/mmap"
)

func fileExists(fname string) bool {
	info, err := os.Stat(fname)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}


func WriteToCSV(csvpath string, dset string, search_type string, num_recs int, bits int, q int, inds[] uint64, searchTime float64, ts time.Time, train float64, wall float64) {
	server, _ := os.Hostname()
	fname := fmt.Sprintf("%salgodirect_gemini_%s_%s_%s_%d_%d_%f_%f.csv", csvpath, server, dset, search_type, num_recs, bits, train, wall)
	if !fileExists(fname) {
		fmt.Println("creating file", fname)
		file, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
		if err != nil {
			panic(err)
		}
		writer := csv.NewWriter(file)
		row := []string{"ts", "q_index", "search_time", "inds" }
		err = writer.Write(row)
		if err != nil {
			panic(err)
		}
		writer.Flush()
	} 
    
	//fmt.Println("appending file", fname)
	file, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		panic(err)
	}

	writer := csv.NewWriter(file)

    inds_str := fmt.Sprintf("%d-%d-%d-%d=%d-%d-%d-%d-%d-%d",
        inds[0], inds[1], inds[2], inds[3], inds[4], inds[5], inds[6], inds[7], inds[8], inds[9] )
	row := []string{
		fmt.Sprintf("%d", time.Now().Unix()),
		fmt.Sprintf("%d", q),
		fmt.Sprintf("%f", searchTime),
		fmt.Sprintf("%s", inds_str),
	}
	err = writer.Write(row)
	if err != nil {
		panic(err)
	}
	writer.Flush()
}

/*
func speed_test(cpus int, queries ) error {
	var times []time.Duration
	m := &sync.Mutex{}
	queues := make([][][]byte, cpus)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < cpus

	return nil
}

func run_queries(queryVectors [][]float32, index *hnsw, k int, ef int) [][]uint64 {
	arr := make([][]uint64, len(queryVectors)) // initialize array for returned indices
	queues := make([][][]byte, CPUs)
	for i := 0; i < len(queryVectors); i++ {
		query := byte(queryVectors[i])
		worker := i % CPUs
		queues[worker] = append(queues[worker], query)
	}

	for i, vec := range queryVectors {
		inds, _, err := index.knnSearchByVector(vec, int(k), ef, nil)
		arr[i] = inds
		if err != nil {
			panic(err)
		}
	}
	return arr
}
*/

func TestBench(t *testing.T) {
    
    fmt.Println("TestBench start...")

    //
    // get config
    //
    dset := os.Getenv("DSET")
    fmt.Println("dset=",dset)
    dim, _  := strconv.Atoi(os.Getenv("DIM"))
    fmt.Println("dim=",dim)
    num_recs, _  := strconv.Atoi(os.Getenv("NUMRECS"))
    fmt.Println("numrecs=",num_recs)
    bits, _ := strconv.Atoi(os.Getenv("BITS"))
    fmt.Println("bits=",bits)
    search_type := os.Getenv("SEARCH")
    fmt.Println("searchtype=",search_type)
    csvpath := os.Getenv("CSVPATH")
    fmt.Println("csvpath=", csvpath)
    
    //
    // check file paths
    //

    // get and check data path
    data_path := os.Getenv("DATAPATH")
    b := fileExists(data_path)
    if !b {
        fmt.Println("data path does not exist", data_path)
        t.FailNow()
    }

    // get, check, and open query path
    query_path := os.Getenv("QUERYPATH")
    b = fileExists(query_path)
    if !b {
        fmt.Println("query does not exist", query_path)
        t.FailNow()
	}
    query_reader, oerr := mmap.Open(query_path)
    if oerr!=nil {
        fmt.Printf("Cannot open file %v\n", query_path)
        t.FailNow()
	}
    defer query_reader.Close()

    //
    // initialize a gemini index config
    //
    geminiConfig := ent.UserConfig{}
    geminiConfig.SetDefaults()
    geminiConfig.SearchType = search_type // via env
    geminiConfig.NBits = bits // via env

	index, err := New(Config{
		RootPath:              "doesnt-matter-as-committlogger-is-mocked-out",
		ID:                    "unittest",
	}, geminiConfig)
    if err!= nil {
	    assert.Nil(t, err)
        t.FailNow()
    }

    // 
    // initiate a batch import from npy file
    //
    wall_start := time.Now()
    berr := index.BatchImport( data_path, dim, num_recs )
    if berr!=nil {
	    assert.Nil(t, berr)
        t.FailNow()
    }

    //
    // loop until async build/train is finished or an error
    //
    train_start := time.Now()
    ok := true
    for true {
        vec := make([]float32, dim)
        _, _, serr := index.SearchByVector(vec, 1, nil)
        if serr!=nil {
            fmt.Println("SearchByVector: ",serr)

            if strings.HasPrefix(fmt.Sprint(serr), "Async index build is in progress") {
                time.Sleep(time.Second)
                continue
            } else if strings.HasPrefix(fmt.Sprint(serr), "Async index build completed.") {
                break
            } else {
                ok = false
                break
            }
        }
    }
    if !ok {
        fmt.Println("index impport/build error.")
        t.FailNow()
    }
    train_time := time.Since(train_start).Seconds()
    wall_time := time.Since(wall_start).Seconds()

    //
    // prepare queries
    //
    queryVectors := make([][]float32, 1000)
    for i := range queryVectors {
        queryVectors[i] = make([]float32, dim)
    }
    _, err = Numpy_read_float32_array(query_reader, queryVectors, int64(dim), int64(0), int64(1000), int64(128))
    if err!=nil {
        assert.Nil(t, err)
        t.FailNow()
    }

    //
	// loop on queries
    //
    for i := 0; i < 1000; i++ {
        tm := time.Now()
        inds, _, serr := index.SearchByVector(queryVectors[i], 10, nil)
		search_time := time.Since(tm).Seconds()
        if serr!=nil {
            assert.Nil(t, serr)
            t.FailNow()
        }
		WriteToCSV( csvpath, dset, search_type, num_recs, bits, i, inds, search_time, tm, train_time, wall_time )
	}

    //
    //  finalize and cleanup
    //
	fmt.Println("Done.")
}
