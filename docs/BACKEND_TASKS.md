# Backend Task Plan

This plan covers the backend and AI tracks of [PLAN.md](PLAN.md). The frontend is already on `main` and runs every screen against an in-browser mock of the API (`npm run dev:mock`). The backend's job is to serve the same contract for real, so that `npm run dev` works with the mock switched off.

## How the work runs

- **One shared branch.** Every backend and AI task (B0 to A3) is built on `m1/backend-foundation` and merges to `main` together at the end, through PR #19. B0 went first and holds everything the other tasks share: settings, the database, all 21 tables in one Alembic migration, every Pydantic schema, the audit and update helpers, the Gemini client, and `main.py` with every router already mounted as an empty stub.
- **Then everything else runs in parallel.** B1 to B6, A1 and A2 can all start at once, each in its own session on the same branch. Each owns its own files (listed per task), so sessions rarely touch the same file. A3 starts after A1.
- **Pushing to the shared branch.** Commit small and often. Before every push, `git fetch origin && git merge origin/m1/backend-foundation`, run `pytest`, then `git push origin m1/backend-foundation`; if the push is rejected, fetch, merge and push again. Never rebase or force-push this branch, and don't open separate pull requests.
- **The contract is fixed.** Response and request shapes are already written in `backend/app/schemas/`, mirroring `frontend/src/types/api.ts`. Use them as they are, with `response_model` on every route. If a shape really has to change, change the schema and `frontend/src/types/api.ts` together, keep it small, and list it in your final report.
- **The mock is the reference implementation.** `frontend/src/mocks/handlers.ts` implements every route; port the section for your task, swapping its keyword matching for Gemini where the plan says so. `frontend/src/mocks/seed.ts` is the seed case. Where this plan and the mock differ, this plan wins.
- **Shared files stay frozen.** Don't edit `app/models/`, `alembic/`, `app/core/`, `app/main.py`, `pyproject.toml` or `tests/conftest.py` and `tests/factories.py` unless your task says so. If you need a new column or helper, ask first; a second migration from the same parent leaves Alembic with two heads.

## Conventions

```python
from fastapi import APIRouter, BackgroundTasks, status

from app.core.deps import ActorId, DbSession
from app.core.records import apply_update, get_or_404, record_audit
from app.models import Case, Person
from app.schemas.people import Person as PersonOut, PersonCreate, PersonUpdate

router = APIRouter(tags=["people"])


@router.post("/cases/{case_id}/persons", response_model=PersonOut, status_code=status.HTTP_201_CREATED)
def create_person(case_id: int, body: PersonCreate, db: DbSession, actor_id: ActorId):
    case = get_or_404(db, Case, case_id)
    person = Person(case_id=case.id, **body.model_dump())
    db.add(person)
    db.flush()
    record_audit(db, actor_id, "modified", "person", person.id, case_id=case.id, detail={"created": True})
    db.commit()
    return person


@router.patch("/persons/{person_id}", response_model=PersonOut)
def update_person(person_id: int, body: PersonUpdate, db: DbSession, actor_id: ActorId):
    person = get_or_404(db, Person, person_id)
    apply_update(db, person, body.model_dump(exclude_unset=True), record_type="person", actor_id=actor_id)
    db.commit()
    return person
```

- **Routes** are plain `def` (FastAPI runs them in a thread pool), use `DbSession` and `ActorId`, and commit once at the end. Paths in the router have no `/api` prefix; `main.py` adds it.
- **Status codes:** 201 on create, 204 on delete, 404 with `"<Record> not found"` from `get_or_404`, and FastAPI's 422 for bad bodies (the schemas already reject unknown fields).
- **Audit:** every create, change, delete, view and export writes an `audit_log` row through `record_audit` or `apply_update`, always with `case_id`, so it shows in `GET /cases/{id}/audit`. Creates log `modified` with `{"created": true}`, deletes log `modified` with `{"deleted": true}`, and review-only changes log `reviewed` (`apply_update` picks this for you).
- **AI rows** are `origin="ai"`, `review_status="pending"`. Rows created through the API are `origin="investigator"`, `review_status="confirmed"` (the model defaults).
- **Deletes** of persons, objects, locations and evidence call `delete_relationships_of`; every delete calls `delete_links_to`.
- **Counts** come from `case_counts`; never store them.
- **Times** are aware UTC `datetime`s. Columns use `UTCDateTime`, which returns UTC and emits `...Z` in JSON.
- **JSON columns:** assign a new list or dict (`stage.event_ids = [*stage.event_ids, 5]`) rather than mutating in place.
- **Gemini** is optional. With no `GEMINI_API_KEY` everything still runs: AI steps are skipped or fall back to simple local logic, and tests never call Gemini.
- **Tests** go in `backend/tests/test_<your router or service>.py`, using the `client` and `db` fixtures and the builders in `tests/factories.py`. Each test gets a fresh database.

## Done when

- Every route in the task behaves like the mock: same shapes, filters, ordering, audit entries and cascades.
- `pytest` passes in `backend/`, with tests for each route and each rule the task lists.
- The matching frontend screens work in `npm run dev` against the real backend, with the seed loaded (once B6 is in).
- The session's final report lists any shared file it touched.

## Phase 0

### B0 Foundation (`m1/backend-foundation`)

Done first, so the other tasks start from it.

- `pyproject.toml` with every backend dependency, including the AI ones, so no later task edits it.
- `core/config.py` (root `.env`, relative paths against `backend/`), `core/db.py` (WAL, foreign keys, `check_same_thread` off), `core/storage.py` (save under `storage/<case_id>/`, SHA-256), `core/deps.py` (`DbSession`, `ActorId` from `X-Investigator-Id` or `?investigator_id=`), `core/records.py` (`get_or_404`, `record_audit`, `apply_update`, `delete_links_to`, `delete_relationships_of`, `case_counts`, `RECORD_MODELS`).
- `models/`: every table in PLAN.md's data model, plus `evidence.processed_at` (for the alerts feed), `evidence_chunks.ordinal`, `ai_runs.case_id` (for case-level runs), `audit_log.case_id` (so the case audit is one query) and `buildings.description`.
- `alembic/versions/..._0001_initial_schema.py`, generated from the models, in batch mode.
- `schemas/`: one module per router, mirroring `frontend/src/types/api.ts`.
- `main.py` with CORS for port 3000, `/api/health` and every router mounted. Each `routers/*.py` is an empty stub naming its task.
- Interface stubs: `services/ai/pipeline.process_evidence(evidence_id)` and `services/search/indexing.index_evidence(db, evidence_id, pages, paged=False)`, plus the shared Gemini client `services/ai/client.get_client()`.
- `tests/conftest.py`, `tests/factories.py` and tests for the helpers.

## Phase 1: parallel

### B1 Cases and people

Owns `routers/cases.py`, `routers/people.py`, `tests/test_cases.py`, `tests/test_people.py`. Unblocks frontend T3 and T4.

- `GET /investigators`, ordered by id.
- `GET /cases?q=&status=&priority=`: newest first, each row with `evidence_count`. `q` matches case number, title, description, lead investigator name, location addresses and person names, case-insensitively.
- `POST /cases`: `case_number` is `FX-<year>-<id padded to 4>`, set after the insert. Reject unknown `lead_investigator_id` and `investigator_ids` with 422.
- `GET`, `PATCH /cases/{id}`: `apply_update`, and bump `updated_at`.
- `GET /cases/{id}/workspace`: the case, its persons, objects, locations, evidence, events, relationships and injuries, every evidence link whose evidence is in the case, and `case_counts`.
- `GET`, `POST /cases/{id}/persons|objects|locations` and `PATCH`, `DELETE /persons|objects|locations/{id}`, as in the mock's `resource()` helper. Deletes remove the record's relationships and links.

### B2 Evidence

Owns `routers/evidence.py`, `services/evidence/`, `tests/test_evidence.py`. Unblocks frontend T5.

- `POST /cases/{id}/evidence`: multipart field `files`, several per request. For each file: save with `core/storage.save_original`, guess `type` from the MIME type as the mock's `evidenceTypeFor` does, title from the file name, status `queued`, audit `uploaded` with the file name and SHA-256, then `background_tasks.add_task(process_evidence, evidence.id)`. Returns 201 with the list of created rows.
- `GET /cases/{id}/evidence` and `GET /evidence?q=&type=` (newest first; `q` matches title, description, case number, case title and summary).
- `GET /evidence/{id}`: the detail with `extracted_text`, `links` and `custody` (audit rows for this evidence, oldest first).
- `PATCH /evidence/{id}`: type, title, description, notes.
- `GET /evidence/{id}/file`: the original as a `FileResponse` with its MIME type, `inline` or, with `?download=1`, `attachment` under its title. Logs `viewed` or `exported`. The investigator comes from `?investigator_id=` here, which `ActorId` already reads.
- `POST /evidence/{id}/reprocess`: back to `queued`, audit, and queue the pipeline again.
- `POST /evidence-links` (check the evidence and the target exist, using `RECORD_MODELS`) and `DELETE /evidence-links/{id}`. Audit on the target record.
- `POST /cases/{id}/verify`: recompute every stored file's SHA-256 with `sha256_of`; a missing file is a mismatch with `actual: null`. Audit `reviewed` on the case.

### B3 Graph, timeline and review

Owns `routers/graph.py`, `routers/timeline.py`, `routers/review.py`, `services/timeline/`, and their tests. Unblocks frontend T7, T8 and T9.

- `GET /cases/{id}/graph`: as the mock's `graphFor`. Every person and object is a node; locations and evidence only when a relationship touches them. Sublabels: person role and condition, object plate or kind, location kind, evidence type. Edges only where both ends are nodes. No positions.
- `POST /cases/{id}/relationships`: both ends must exist in this case; an empty `label` becomes the humanised kind ("involved_in" becomes "Involved in"). `PATCH`, `DELETE /relationships/{id}`.
- `GET /cases/{id}/events?category=`: by `time_start`, untimed events last. `POST`, `PATCH`, `DELETE /events/{id}`; `location_id` must belong to the case. Deleting an event also removes it from every stage's `event_ids`.
- `GET /cases/{id}/observations?status=`, `PATCH /observations/{id}`, `GET /cases/{id}/conflicts?status=`, `PATCH /conflicts/{id}`.

### B4 Reconstruction and injuries

Owns `routers/reconstruction.py`, `routers/injuries.py`, `routers/buildings.py`, `services/reconstruction/`, and their tests. Unblocks frontend T10, the 3D track and T13.

- `GET`, `POST /cases/{id}/reconstructions`. A new scenario starts at version 1.
- `GET /reconstructions/{id}`: stages in `stage_order`, each with its scene objects, plus annotations.
- `POST /reconstructions/{id}/stages`: `stage_order` defaults to after the last stage; `event_ids` must be events of the same case.
- `PUT /stages/{id}/scene-objects`: replaces the stage's placements with the list sent; `object_id` must be an object of the same case. Returns the new list.
- `GET`, `POST /reconstructions/{id}/annotations`; `stage_id` must belong to the reconstruction.
- `GET`, `POST /cases/{id}/injuries` and `PATCH /injuries/{id}` (added to the contract by the frontend, for reviewing AI-proposed injuries).
- `GET /buildings/{id}`: the row, or 404 until M6 adds buildings.

### B5 Reports, audit and dashboard

Owns `routers/reports.py`, `routers/dashboard.py`, `services/reports/`, and their tests. Unblocks frontend T11 and T12.

- `POST /cases/{id}/reports`: assemble `ReportContent` as the mock's `buildReport` does, per template (`court`, `summary`, `custody`, `medical`) and options. `sha256` is the SHA-256 of the content as canonical JSON (`json.dumps(content, sort_keys=True, separators=(",", ":"))`). Store it, audit `exported`, return 201.
- `GET /cases/{id}/reports` (newest first, without content; added to the contract by the frontend) and `GET /reports/{id}`.
- `GET /cases/{id}/audit`: the case's audit rows by `audit_log.case_id`, newest first.
- `GET /dashboard`: totals, active (not closed), pending reviews across cases, storage bytes, status counts with every status present, incident points from `incident_site` locations that have coordinates, and investigators with solved (lead on a closed case) and active (on the team of an open case) counts.
- `GET /alerts`: open conflicts (warning), cases with pending AI reviews (info), and evidence that finished or failed processing, from `processed_at` (info or critical). Newest first; ids as in the mock (`conflict-3`, `review-1`, `processing-7-<time>`).

### B6 Seed case

Owns `app/seed.py`, `backend/seed_data/`, `tests/test_seed.py`. Unblocks running the real backend with data.

- Port `frontend/src/mocks/seed.ts` and `files.ts`: the investigators, the four cases, and the full road-accident case (persons, vehicles and phone, locations, events, relationships, observations, the signal-colour conflict, injuries, both reconstruction scenarios with Approach and Impact stages and their scene objects, evidence links and audit history).
- Evidence files are real: generate them (PyMuPDF can write the PDFs) or commit small source files to `seed_data/`, then store them with `core/storage.save_bytes`, so hashes and previews are real and `verify` passes. Store the extracted text and call `index_evidence`, so search works without Gemini. Link passages to their chunk.
- `python -m app.seed` loads into an empty database and refuses if cases exist; `python -m app.seed --reset` drops and recreates every table first.

### A1 Evidence AI pipeline

Owns `services/ai/` (except `client.py` and `conflicts.py`), `tests/test_ai_pipeline.py`. Implements PLAN.md's pipeline steps 2 and 3.

- Fill in `process_evidence(evidence_id)`, keeping its signature: status `processing`, extract text, structure it, index it, then `done` with `processed_at`; on any error, `failed` with the reason in `summary` and `processed_at` set.
- **Extract:** PyMuPDF for PDFs (one string per page; fall back to OCR for pages with no text layer), `python-docx` for Word, plain text as is. Gemini OCR for images with text, Gemini transcription for audio and the soundtrack of video. Store `extracted_text`.
- **Structure:** one Gemini call with a JSON response schema: a summary, plus persons, objects, locations, events, relationships and observations, each with the exact passage it came from. Save the raw output to `ai_runs`, the summary on the evidence, and each item as an `origin="ai"`, `review_status="pending"` row with an `evidence_links` row pointing at its passage and chunk. Match people and vehicles that already exist in the case (by name or plate) instead of duplicating them.
- **Index:** call `services/search/indexing.index_evidence` before linking passages, so links can carry chunk IDs.
- Without `GEMINI_API_KEY`: extract what can be read locally, index it, leave the summary empty and finish as `done`.
- Use `services/ai/client.get_client()` (from B0) for every Gemini call; it returns None without a key. Tests patch it.

### A2 Search and cited answers

Owns `services/search/`, `routers/search.py`, `tests/test_search.py`. Implements steps 4 and 5. Unblocks frontend T6.

- Replace the body of `index_evidence`, keeping its signature: page-sized chunks split on paragraph boundaries, embedded with `GEMINI_EMBED_MODEL` in batches and stored as float32 bytes.
- Ranking: cosine similarity with numpy over this case's chunks. Chunks without an embedding (seed data, no key) are embedded on first use when a key is set; without a key, rank by keyword overlap as the mock does.
- `GET /cases/{id}/search?q=`: ranked `SearchHit`s.
- `POST /cases/{id}/ask`: embed the question with the previous turns, take the top chunks, and ask Gemini for JSON `{answer, chunk_ids, follow_ups}` using only those chunks. Return citations for the chunk IDs it used, at most three follow-ups, and log the call to `ai_runs`. Without a key, return the mock's keyword answer.
- Use `services/ai/client.get_client()` (from B0) for every Gemini call.

## Phase 2

### A3 Conflict detection

Starts after A1. Owns `services/ai/conflicts.py`, `tests/test_conflicts.py`. Implements step 6.

- Group a case's observations by subject, place and time, ask Gemini which statements conflict or what is missing, and save `conflicts` rows that list their observation IDs. Skip groups that already have a conflict with the same observations. Log the call to `ai_runs`.
- Called at the end of `process_evidence`, so a new statement can open a conflict, which then shows in the review queue and alerts.

### B7 Integration

After B1 to B6. Run the seed, start both apps with `npm run dev`, and walk PLAN.md's nine MVP steps with the mock off. Fix backend mismatches; hand frontend ones to the frontend's T14 (which runs `npm run gen:api`).

## Running more than one session

Every session works on `m1/backend-foundation`.

| Session | Task |
| --- | --- |
| 1 | B1 Cases and people |
| 2 | B2 Evidence |
| 3 | B6 Seed |
| 4 | A1 AI pipeline, then A3 Conflict detection |
| 5 | A2 Search |
| 6 | B3 Graph, timeline, review |
| 7 | B4 Reconstruction, then B5 Reports and dashboard |
| 8 | B7 Integration, once the others are in |

With fewer sessions, take the tasks in this order: B1, B2, B6, A1, A2, B3, B4, B5, A3. B1, B2 and B6 together give a working case and evidence flow on the real backend.
