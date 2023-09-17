// RUN: tree-opt %s | tree-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = tree.foo %{{.*}} : i32
        %res = tree.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @tree_types(%arg0: !tree.custom<"10">)
    func.func @tree_types(%arg0: !tree.custom<"10">) {
        return
    }
}