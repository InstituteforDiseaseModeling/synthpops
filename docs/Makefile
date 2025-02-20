# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -jauto
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

.PHONY: clean
clean: # The -f is to ignore missing files, not to force remove
	rm -rf $(BUILDDIR)
	rm -f synthpops*.rst
	rm -f modules.rst

.PHONY: generate-api

# Generates the API doc
generate-api:
	sphinx-apidoc -f -e -M -o . ../synthpops
	mv synthpops.rst modules.rst

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
