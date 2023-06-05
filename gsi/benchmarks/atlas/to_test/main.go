package main
import (
    "encoding/binary"
    "fmt"
    "log"
    "math"
    "golang.org/x/exp/mmap"
)
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
func main() {
    readerAt, err := mmap.Open("/mnt/nas1/atlas_data/benchmarking/atlas_test_0531.npy")
    if err != nil {
        log.Fatal(err)
    }
    var dim, size int64 = 768, 251084
    arr := make([][]float32, size)
    for i := range arr {
        arr[i] = make([]float32, dim)
    }
    _, aerr := Numpy_read_float32_array(readerAt, arr, dim, int64(0), size, int64(0))
    if aerr != nil {
        log.Fatal(aerr)
    }
    fmt.Println(len(arr), len(arr[0]), arr[len(arr)-1])
}