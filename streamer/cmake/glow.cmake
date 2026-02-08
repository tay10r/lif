include_guard()

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

FetchContent_Declare(glow
  URL "https://github.com/tay10r/glow/archive/refs/tags/v0.3.0.zip"
  URL_HASH "SHA256=3ea2705da47e9d415f08df1aa1cb69568009a95aefc6814c63bcd50b93e068a0"
)

FetchContent_MakeAvailable(glow)
