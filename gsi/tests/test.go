
package main

import (
    "fmt"
    "os"
    "log"
    "io/ioutil"
    //"testing"

    geminiplugin "github.com/gsi/weaviate/gemini_plugin"
)

func main() {

    // Create a temp file that will automatically delete itself when this app is done
    file, err := ioutil.TempFile("/tmp/", "gemini_plugin_test_")
    if err != nil {
        log.Fatal(err)
    }
    defer os.Remove(file.Name())
    fmt.Println("Got temp file->", file.Name())

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
    row_count, dim, aerr := geminiplugin.Numpy_append_float32_array( file.Name(), arr, 96, 1)
    if aerr != nil {
        fmt.Println("Error creating numpy file with float32 array.", aerr)
    } else {
        fmt.Println("Got return data", row_count, dim)
    }

    fmt.Println("Done.")
}
