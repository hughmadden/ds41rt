# DS41RT operation manual

## Development and numbered releases

Develop and apply documentation changes on `dev`. Keep `main` at the latest
qualified open-source release. Preserve unrelated local changes when switching
branches, and use isolated checkouts for builds that must freeze source.

Before promotion, complete [the release checklist](release-v1-checklist.md),
validate the final amd64 coordinator and arm64 worker container pair on the
five-host deployment, and publish those exact images to GitHub's container
registry. Record source revisions, dependency pins, image digests, launch
configuration and qualification evidence. Complete intended documentation
updates on `dev` before advancing release branches.

Once those gates pass, advance `main` to the qualified `dev` head. Normally
there are no main-only changes, so use a fast-forward (`git merge --ff-only dev`
while on main). If main has unique commits, inspect them and rebase main onto
dev as requested by the release workflow; qualify the resulting source again
if it differs from the validated image source. Do not discard unique commits
or force-push shared history without resolving the divergence explicitly.

Create `release/vX` at the qualified release commit, substituting the actual
numbered release for X, and publish main and that branch. Continue subsequent
development on `dev`. Never advance main or create a release branch simply
because a development build passes a narrow check.

The repository owner makes the published container images public manually.
Include that remaining owner action in the final release handoff, after all
other requested release work is complete.

## Build and serving references

Use [DEVELOPER.md](../DEVELOPER.md) for source and dependency verification and
[the container guide](../docker/README.md) for build/run commands. Those commands
remain subject to the first-release script qualification gate; historical
development launch arrays do not prove reproducible release operation.
