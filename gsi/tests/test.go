
package main

import (
    "fmt"
    "time"
    "math/rand"
    "log"
    "os"

    geminiplugin "github.com/gsi/weaviate/gemini_plugin"
)

// Generate a random string useful for generate a temp filename that does not already exist
func randomString(length int) string {
    rand.Seed(time.Now().UnixNano())
    b := make([]byte, length+2)
    rand.Read(b)
    return fmt.Sprintf("%x", b)[2 : length+2]
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
    arr := make([][]float32,1 )
    arr[0] = make([]float32, 96 )
    fmt.Println(arr, arr[0])

    //
    // Build a numpy file with the array
    //
    fmt.Println("Testing Numpy_append_float32_array")
    row_count, dim, aerr := geminiplugin.Numpy_append_float32_array( ranfilepath, arr, 96, 1)
    if aerr != nil {
        fmt.Println("Error creating numpy file with float32 array.", aerr)
    } else {
        fmt.Println("Got return data", row_count, dim)
    }

    fmt.Println("Done.")
}
