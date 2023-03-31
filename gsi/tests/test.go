
package main

import (
    "fmt"
)

func testa( v int ) int {

    if v<0 {
        fmt.Println("less than zero")
    } else {
        fmt.Println("not less than zero")
    }
    return 0
}

func main() {
    testa(-1)
    testa(0)
}
