#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
  CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"
PROJECT_DIR="$(
  CDPATH= cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1
  pwd -P
)"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [R_LIBRARY_DIRECTORY]" >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  R_LIBRARY=$1
elif [[ -n "${R_LIBS_USER:-}" ]]; then
  R_LIBRARY=$R_LIBS_USER
else
  R_LIBRARY="${PROJECT_DIR}/.r-library"
fi

if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript was not found. Install R before running this script." >&2
  exit 1
fi

mkdir -p -- "${R_LIBRARY}"
R_LIBRARY="$(
  CDPATH= cd -- "${R_LIBRARY}" >/dev/null 2>&1
  pwd -P
)"
export R_LIBS_USER="${R_LIBRARY}"

echo "Installing R packages into: ${R_LIBS_USER}"

Rscript --vanilla -e '
library_path <- Sys.getenv("R_LIBS_USER")
.libPaths(c(library_path, .libPaths()))

packages <- c("chron", "evd", "ensembleBMA", "ensembleMOS")
installed_locally <- rownames(
  installed.packages(lib.loc = library_path, noCache = TRUE)
)
missing <- setdiff(packages, installed_locally)

if (length(missing) > 0) {
  install.packages(
    missing,
    lib = library_path,
    repos = "https://cloud.r-project.org",
    dependencies = FALSE
  )
} else {
  message("All required R packages are already installed.")
}

failed <- packages[
  !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(failed) > 0) {
  stop(
    "Required R packages are still unavailable: ",
    paste(failed, collapse = ", ")
  )
}

for (package in packages) {
  cat(package, as.character(packageVersion(package)), "\n")
}
'

echo
echo "R package installation complete."
echo "Use this environment when running the application:"
echo "export R_LIBS_USER=\"${R_LIBS_USER}\""
