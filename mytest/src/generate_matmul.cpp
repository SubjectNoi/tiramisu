#include "tiramisu/tiramisu.h"
#include "Halide.h"
#include <iostream>
#define M 1024
#define N 1024
#define K 1024
using namespace std;
using namespace tiramisu;

int main(int argc, char** argv) {
	tiramisu::init("matmul_1");
	constant _m("M", expr((int32_t) M));
	constant _n("N", expr((int32_t) N));
	constant _k("K", expr((int32_t) K));
	var i("i", 0, _m), j("j", 0, _n), k("k", 0, _k);
	input A("A", {"i", "j"}, {M, K}, p_uint8);
	input B("B", {"i", "j"}, {K, N}, p_uint8);
	computation C_init("C_init", {i, j}, expr((uint8_t) 0));
	computation C("C", {i, j, k}, p_uint8);
	C.set_expression(C(i, j, k - 1) + A(i, k) * B(k, j));

	var i0("i0"), j0("j0"), i1("i1"), j1("j1");
	C_init.tile(i, j, 64, 64, i0, j0, i1, j1);
	C.tile(i, j, 64, 64, i0, j0, i1, j1);
	
	C.parallelize(i0);
	
	C.vectorize(j1, 32);

	// C.after(C_init, j);

	buffer b_A("b_A", {expr(M), expr(K)}, p_uint8, a_input);
	buffer b_B("b_B", {expr(K), expr(N)}, p_uint8, a_input);
	buffer b_C("b_C", {expr(M), expr(N)}, p_uint8, a_output);
	
	A.store_in(&b_A);
	B.store_in(&b_B);
	C_init.store_in(&b_C, {i, j});
	C.store_in(&b_C, {i, j});

	tiramisu::codegen({&b_A, &b_B, &b_C}, "matmul_1.o");
	return 0;
}
