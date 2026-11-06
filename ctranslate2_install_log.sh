#!/data/data/com.termux/files/usr/bin/bash
# =============================================================================
# ct2_install_log.sh
# CTranslate2 在 Termux 上的完整安装记录（复盘 + 可复用脚本）
# 设备：aarch64 / Android / Termux / Python 3.14 / Clang 21
# =============================================================================
# 说明：
#   本脚本前半部分是「踩坑记录」，只打印不执行；
#   后半部分「最终可用流程」是真正会跑的命令。
#   如果你想直接重装，跳到 "PART 2" 执行即可。
# =============================================================================
set -euo pipefail

SRC_DIR="$HOME/Python/CTranslate2"
BUILD_DIR="$SRC_DIR/build"
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"

# =============================================================================
# PART 1 — 踩坑记录（只打印，不执行）
# =============================================================================
cat <<'EOF'
================================================================================
PART 1 — 踩坑记录（从最开始到现在）
================================================================================

【坑 0】在源码根目录直接 pip install .
  现象：ERROR: Directory '.' is not installable.
        Neither 'setup.py' nor 'pyproject.toml' found.
  原因：CTranslate2 根目录是 C++ 项目，Python 打包文件在 python/ 子目录。
  结论：应 cd python 后再 pip install；但 Termux 上源码编译依赖多，不推荐。

【坑 1】pip 解析 termux-user-repository.github.io 失败
  现象：NameResolutionError / [Errno 7] No address associated with hostname
  原因：网络/DNS 只能访问部分域名（cqupt、清华等通，termux.dev、TUR 不通）。
  结论：改用能通的镜像；或直接用 pip install ctranslate2 装预编译包。

【坑 2】cmake 报 OpenBLAS include directory not found
  现象：CMake Error at CMakeLists.txt:460: OpenBLAS include directory not found
  原因：只写了 -DWITH_OPENBLAS=ON，没告诉 CMake 头文件在哪。
  解法：加 -DOPENBLAS_INCLUDE_DIR=$PREFIX/include

【坑 3】make 报 make: *** No targets specified and no makefile found.
  原因：上一步 cmake 配置失败，没生成 Makefile。
  结论：cmake 成功（Configuring done / Generating done）后再 make。

【坑 4】编译报 fatal error: 'cblas.h' file not found
  原因：Termux 的 OpenBLAS 头文件在 $PREFIX/include/openblas/ 子目录，
        编译器默认不去子目录找。
  解法：export CXXFLAGS=" -I${PREFIX}/include/openblas -D_GNU_SOURCE"

【坑 5】apt install openblas 失败（DNS + dpkg 状态混乱）
  现象：Something wicked happened resolving 'tur.kcubeterm.com'
        E: Sub-process dpkg exited unexpectedly
  原因：TUR 源域名解析失败；之前 apt 折腾导致 dpkg 数据库状态异常。
  结论：能连 cqupt/清华/中科大源；必要时精简 apt 源，或改用 pip 预编译包。

【坑 6】链接 ct2-translator 报
        undefined symbol: std::__ndk1::__hash_memory(void const*, unsigned long)
  原因：-lc++_shared 已进链接命令，但链接器用 -l 搜索时拿到的
        不是验证过含该符号的那个 libc++_shared.so（路径优先级/缓存问题）。
  验证：nm -D $PREFIX/lib/libc++_shared.so | grep __hash_memory
        有输出 => 库本身没问题，是链接器找错了。
  弯路：export LDFLAGS 后直接 make —— 无效，Makefile 已生成，不重读环境变量。
        必须 rm -rf * 后重新 cmake。

【最终解法】强制链接器用绝对路径的 libc++_shared.so：
  -DCMAKE_EXE_LINKER_FLAGS="${PREFIX}/lib/libc++_shared.so -landroid-posix-semaphore"
  结果：libctranslate2.so、头文件、ct2-translator 全部构建并安装成功。

【补充】相对链接方案（本脚本 PART 3 提供）：
  把 libc++_shared.so 复制到 build/，用 -L.. -lc++_shared + $ORIGIN RPATH
  重新链接 ct2-translator，实现构建目录内自包含。

================================================================================
EOF

# =============================================================================
# PART 2 — 最终可用流程（会真正执行）
# =============================================================================
echo "==> PART 2: 开始最终可用流程"

echo "==> 2.1 检查 OpenBLAS 头文件"
if [ ! -f "$PREFIX/include/openblas/cblas.h" ]; then
  echo "❌ 缺少 $PREFIX/include/openblas/cblas.h"
  echo "   请先安装 openblas 开发包（能通的源），或确认头文件路径。"
  exit 1
fi
echo "✅ cblas.h 存在"

echo "==> 2.2 清理并重建 build 目录"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "==> 2.3 设置编译环境变量"
export CXXFLAGS=" -I${PREFIX}/include/openblas -D_GNU_SOURCE"

echo "==> 2.4 cmake 配置（关键：EXE_LINKER_FLAGS 用绝对路径的 libc++_shared.so）"
cmake "$SRC_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_MKL=OFF \
  -DWITH_OPENBLAS=ON \
  -DOPENBLAS_INCLUDE_DIR="$PREFIX/include" \
  -DOPENMP_RUNTIME=NONE \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_EXE_LINKER_FLAGS="${PREFIX}/lib/libc++_shared.so -landroid-posix-semaphore"

echo "==> 2.5 验证链接参数已写入"
grep "c++_shared" cli/CMakeFiles/translator.dir/link.txt || {
  echo "⚠️  link.txt 里没看到 c++_shared，检查 cmake 输出"; exit 1;
}

echo "==> 2.6 编译并安装"
make install -j"$(nproc)"

echo "==> 2.7 验证 C++ 核心库安装"
ls -l "$PREFIX/lib/libctranslate2.so" "$PREFIX/include/ctranslate2/generator.h"
ls -l "$PREFIX/bin/ct2-translator" 2>/dev/null || echo "（ct2-translator 未装到 bin，可忽略）"

echo "✅ PART 2 完成：C++ 核心库 + 头文件 + CLI 已安装"

# =============================================================================
# PART 3 — 可选：相对链接重新生成 ct2-translator（自包含 build 树）
# =============================================================================
echo "==> PART 3: 可选，相对链接 ct2-translator"
read -r -p "是否执行相对链接重生成 ct2-translator？[y/N] " ans || true
if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
  cd "$BUILD_DIR"
  cp -f "$PREFIX/lib/libc++_shared.so" "$BUILD_DIR/libc++_shared.so"
  cd "$BUILD_DIR/cli"
  "${CXX:-/data/data/com.termux/files/usr/bin/c++}" \
    -I"$PREFIX/include/openblas" -D_GNU_SOURCE \
    -Wall -Wextra -O3 -DNDEBUG \
    -L.. -lc++_shared -landroid-posix-semaphore \
    -Xlinker --dependency-file=CMakeFiles/translator.dir/link.d \
    CMakeFiles/translator.dir/translator.cc.o \
    -o ct2-translator \
    -Wl,-rpath,'$ORIGIN/..' \
    -Wl,-rpath,'$ORIGIN' \
    ../libctranslate2.so.4.8.2
  echo "--- readelf ---"
  readelf -d ct2-translator | grep -E "NEEDED|RPATH|RUNPATH" || true
  echo "--- ldd ---"
  ldd ct2-translator | grep -E "c\+\+|ctranslate2" || true
  echo "✅ 相对链接完成：$BUILD_DIR/cli/ct2-translator"
else
  echo "跳过 PART 3"
fi

# =============================================================================
# PART 4 — 编译 Python 绑定（最后一步）
# =============================================================================
echo "==> PART 4: 编译 Python 绑定"
read -r -p "是否继续编译 Python 绑定？[y/N] " ans2 || true
if [[ "${ans2:-N}" =~ ^[Yy]$ ]]; then
  cd "$SRC_DIR/python"
  export CTRANSLATE2_ROOT="$PREFIX"
  pip install .
  echo "==> 验证 import"
  python -c "import ctranslate2; print('ctranslate2 version:', ctranslate2.__version__)"
fi

echo "🎉 全部流程结束。"
EOF
