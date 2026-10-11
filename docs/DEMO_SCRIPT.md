# Evidentia demo video script

**Length:** about 3 minutes (about 430 spoken words at a calm 140 words per minute)
**Format:** screen recording with voiceover, 1080p, no face cam needed
**Story:** one road-accident case, followed from the first upload to the final report
**Tagline:** *Every finding points back to its evidence.*

Before you record, check each scene against the running build. The script follows the nine MVP steps in `docs/PLAN.md`. If a step isn't working yet, cut that scene and keep the rest.

## Setup checklist

- Backend and frontend running, database freshly seeded: `python -m app.seed --reset`
- Browser at 1920×1080, zoom 100%, bookmarks bar hidden, notifications off
- Start on the Dashboard, signed in as the lead investigator
- Have one extra evidence file ready to drag in (a witness statement PDF works best)
- Pre-warm the Copilot with one throwaway question so the first answer isn't slow
- Record the voiceover separately and lay it over the screen capture. Speed up any wait over 2 seconds.

## Script

| # | Time | On screen | Voiceover |
| --- | --- | --- | --- |
| 1 | 0:00–0:15 | Black screen, then a fast montage: a crash photo, a witness statement, a dashcam still, a phone log. The Evidentia logo fades in. | "After a road accident, the truth is scattered across statements, photos, videos and logs. Investigators spend days piecing it together, and one missed detail can change the outcome. This is Evidentia, an AI forensic investigation platform." |
| 2 | 0:15–0:30 | Dashboard: case counts, incident map, alerts feed. Hover the alerts bell. | "Evidentia opens on a live view of every active case: what's new, what needs review, and where incidents happened." |
| 3 | 0:30–0:50 | Click **New case**. Type a title, set the incident type to road accident, and assign the lead. Save. The workspace opens. | "Let's open a new road-accident case. The officer enters only what they know, and a workspace is ready in seconds." |
| 4 | 0:50–1:10 | Drag the witness statement PDF into the evidence locker. Show the status change from *queued* to *processed*. Open the evidence panel: SHA-256 hash, AI summary, custody ledger. | "Evidence goes in with one drag. Evidentia stores the original, computes a SHA-256 hash, logs who touched it, and reads it with AI to extract people, vehicles, places and events. The original file is never altered." |
| 5 | 1:10–1:35 | Open the **Copilot**. Type: *"Which driver had a green light?"* The answer appears with citation chips. Click a chip: the source passage opens next to the original file. | "Now the investigator can simply ask. Every answer cites the exact passage it came from. Click a citation and you're looking at the source. No black box, and nothing to take on faith." |
| 6 | 1:35–1:55 | Open the **Review queue**. Show a conflict card with two statements that disagree. Confirm one AI finding and reject another. | "The AI proposes, the investigator decides. Extracted facts wait in a review queue, and conflicting statements are flagged automatically. Nothing becomes a finding until a human confirms it." |
| 7 | 1:55–2:15 | Open the **Evidence canvas**. Nodes appear for the people, vehicles and evidence. Click a link between a vehicle and a witness to show its label and source. | "Evidentia maps how everything connects: drivers, vehicles, witnesses, and the evidence behind each link." |
| 8 | 2:15–2:30 | Open the **Timeline**. Scrub through the events. Click one event to jump to its source evidence. Filter by category. | "The timeline orders events by time, and shows whether each is established, disputed or only a hypothesis. Click any event to see its evidence." |
| 9 | 2:30–2:55 | Open the **3D crime scene**. Orbit the intersection. Click a vehicle and show its details and linked evidence. Switch from stage 1 to stage 2 and the vehicles move. Use the measurement tool between the two vehicles. | "Then step into the scene. The accident is rebuilt in 3D from the case records. Select any object to see the evidence behind it, measure distances, and switch between reconstruction stages to compare how the collision may have unfolded." |
| 10 | 2:55–3:15 | Open **Reports**. Show the summary: evidence inventory with hashes, timeline, confirmed findings, open conflicts. Click **Verify integrity** and show it pass. | "Finally, a complete investigation report is generated from the same records: the evidence with hashes, the timeline, confirmed findings, and every unresolved conflict. A single click re-checks that no file has been tampered with." |
| 11 | 3:15–3:30 | Pull back to the workspace, then to the logo and tagline. | "From scattered files to a traceable, explainable investigation. Evidentia: every finding points back to its evidence." |

## Shorter version (60 seconds, for social or the pitch round)

Keep scenes 1, 4, 5, 7, 9 and 11. Trim each voiceover to one sentence:

1. "Accident evidence is scattered across files, and piecing it together takes days."
2. "Evidentia reads each upload, hashes it, and logs its custody."
3. "Ask a question and get an answer that cites its source."
4. "See how people, vehicles and evidence connect."
5. "Step into a 3D reconstruction and compare how events may have unfolded."
6. "Every finding points back to its evidence."

## Recording tips

- Rehearse the click path twice. Cursor wandering is the biggest giveaway of an unrehearsed demo.
- Zoom the browser to 110–125% for any panel with small text, such as citations and the custody ledger.
- Pause for one second on each key result (the citation, the conflict card, the stage switch) so viewers can read it.
- If the Gemini key is missing, the Copilot and extraction steps won't work. Check that before recording, or use `npm run dev:mock` and avoid claiming live AI in scenes 4–6.
- Add light background music at about -20 dB and captions for the voiceover.
- Don't show `.env`, API keys, or the terminal in frame.

## If something isn't ready

| Scene | Fallback |
| --- | --- |
| 5 Copilot | Show the cited search results instead of a written answer |
| 6 Review queue | Cut it and move its line to scene 4 |
| 9 Second stage | Stay on stage 1 and show object selection and evidence markers only |
| 10 Verify integrity | Show the evidence inventory with hashes only |
