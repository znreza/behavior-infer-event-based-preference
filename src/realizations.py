"""Three paraphrases per anchor event, keyed by "attribute/event_id".

Attribute-scoped so that ids reused across attributes cannot collide.

Rules followed throughout: first person, describes WHAT HAPPENED and never what
the person now wants, never names any state label of its attribute, one causal
step. `check_leaks` in generate_v2.py enforces the third rule mechanically.
"""

R = {

# ---------------------------------------------------------------- transport
"transport/cost": [
 "The insurance renewal came in higher than my rent, and that is before the permit and two repairs I keep putting off.",
 "I totted up what the car costs me monthly and it is more than I clear from the overtime I took on to pay for it.",
 "Between the garage bill and the residents' permit going up again, I cannot justify what is parked outside.",
],
"transport/nearby": [
 "We moved in February and now work, the shops and the surgery are all within about two kilometres.",
 "The new flat is close enough to everything that the longest trip I make in a day is about fifteen minutes.",
 "Since the move nothing I need is further than the end of the high street.",
],
"transport/injury": [
 "I fractured my wrist in November and cannot put any weight through it for months.",
 "The consultant says the wrist will not take load until spring at the earliest.",
 "My wrist is in a cast and I have been told not to grip or lean on anything.",
],
"transport/safety": [
 "They have dug up the whole protected route and pushed everything onto the narrow roads with the lorries.",
 "The separated path closed for the bridge works and the diversion runs along a fast dual carriageway.",
 "Since the roadworks started there is no safe way through; the only route is shared with heavy traffic.",
],
"transport/duty": [
 "The new role has me at four sites a day carrying equipment, none of them near a station.",
 "I now have to move kit between depots on a schedule the rail network does not reach.",
 "The job changed in March and I am out at rural sites all day with tools to bring.",
],
"transport/strike": [
 "The operator dispute has cancelled most services indefinitely and there is no date for resuming.",
 "Nothing is running to timetable since the dispute began, and my distances are short anyway.",
 "The strike has taken out almost the whole timetable for the foreseeable future.",
],

# ---------------------------------------------------------------- exercise
"exercise/knee": [
 "I tore my meniscus on a trail in October and the physio has ruled out anything with impact.",
 "The knee went on a descent, and the surgeon was clear that repeated pounding is what sets it back.",
 "Since the meniscus tear, anything with landing in it leaves the joint swollen for days.",
],
"exercise/heat": [
 "We moved somewhere that sits above forty degrees for four months, but there is a covered facility nearby.",
 "The summer here makes any outdoor exertion dangerous, though the indoor centre is five minutes away.",
 "It is too hot to be outside for most of the year now, and the indoor option is close.",
],
"exercise/race": [
 "I have signed up for an event in September that needs progressively longer distances on foot.",
 "The training plan I committed to builds up mileage on the ground week by week.",
 "I entered a long-distance event, and preparation is all about time on the legs.",
],
"exercise/shoulder": [
 "I did the rotator cuff in the summer and overhead movement is out for at least six months.",
 "The specialist was blunt: no repeated overhead motion until the cuff has healed.",
 "Raising my arm above shoulder height still catches, and I have been told to keep it down.",
],
"exercise/poolclosure": [
 "The aquatic centre closed indefinitely for structural work and the network of lanes and paths is fine.",
 "They shut the facility in March with no reopening date, though the routes around here are safe.",
 "The only place of that kind locally has closed for good.",
],
"exercise/commute": [
 "The rota change means there is no separate slot any more; whatever I do has to double as getting to work.",
 "I have lost the free hour, so training only happens if it is also the journey in.",
 "There is no time set aside now, but the way to work is long and flat.",
],
"exercise/saddle": [
 "A persistent pelvic injury means prolonged pressure there is off the table, though non-weight-bearing effort is fine.",
 "The clinic ruled out anything that puts me seated under load for long, but said unloaded exertion is allowed.",
 "I cannot spend time seated on hard support any more; anything that takes the weight off is fine.",
],
"exercise/hills": [
 "We moved into steep narrow lanes with no shoulder, though footpaths run in every direction.",
 "The roads here are single-track and blind, but there are tracks straight off the doorstep.",
 "Nothing here has room to ride safely; the paths, though, go on for miles.",
],

# ---------------------------------------------------------------- replylength
"replylength/dispatch": [
 "I read these between calls now and usually have about twenty seconds before the next one.",
 "My shift is back-to-back callouts, so anything you send gets looked at while I walk to the vehicle.",
 "There is rarely half a minute between jobs, and that is when I check what has come in.",
],
"replylength/handover": [
 "Whatever you send goes straight into the handover the next shift reads cold.",
 "I paste these onto the board the incoming team scans before they start.",
 "These get passed to people who were not part of the conversation and have to follow them alone.",
],
"replylength/audit": [
 "I moved onto the regulated side in March, where every decision is written up and defended months later.",
 "The new team is audited quarterly and anything I act on needs the reasoning recorded at the time.",
 "Since the transfer nothing counts unless the working behind it can be reconstructed.",
],
"replylength/phone": [
 "I lost desktop access and now check everything on a handset while moving between places.",
 "It is all on a small screen in short bursts now; I do not sit down to read any more.",
 "I gave up the workstation, so this is read standing up on a phone between things.",
],
"replylength/training": [
 "These go to new starters now, who need the steps separated clearly enough to follow.",
 "I reuse what you send as teaching material for people learning the process.",
 "Trainees work from these, so each step has to stand on its own.",
],
"replylength/dispute": [
 "There is a formal challenge running and every recommendation has to have its full reasoning set out.",
 "The review board rejected the summary version and asked for the complete argument behind each point.",
 "A complaint means summaries are no longer enough; the whole line of reasoning has to be there.",
],

# ---------------------------------------------------------------- replytone
"replytone/bereave": [
 "I lost my mother three weeks ago and I am not coping well with things put flatly.",
 "There was a sudden death in the family and I am finding anything delivered without care hard to take.",
 "Since the funeral I have not been able to absorb things said starkly.",
],
"replytone/shared": [
 "These get pasted into a channel the whole team reads, and any judgement in them lands on colleagues.",
 "What you send is now read by the people it is about, so anything evaluative reads as a verdict on them.",
 "I share these openly now; anything that sounds like an assessment of someone gets taken personally.",
],
"replytone/missed": [
 "A problem got described so kindly last quarter that nobody realised how bad it was until it cost us.",
 "The last review buried a serious fault in encouragement and we found out too late.",
 "Something important was softened to the point that I did not act on it, and it became expensive.",
],
"replytone/clinical": [
 "I am drafting patient-facing material where anything that sounds like comfort has clinical implications.",
 "The regulator treats reassuring phrasing in this material as a claim, so it has to be stripped out.",
 "This goes into medical documentation where encouragement would be read as a promise.",
],
"replytone/solo": [
 "The reorganisation left me the only person on this, with nobody to catch it if I get something wrong.",
 "Everyone else moved off the project, so there is no second pair of eyes on any decision I make.",
 "I am on my own with this now and nothing I do gets checked before it ships.",
],
"replytone/relapse": [
 "I have had a bad relapse and flat delivery is landing as though nobody cares.",
 "My health has gone downhill and affectless replies feel dismissive at the moment.",
 "I am in a difficult stretch and things said without any warmth are hard to sit with.",
],

# ---------------------------------------------------------------- replyformat
"replyformat/scan": [
 "I read these on a handset while moving between sites and need to take them in at a glance.",
 "This gets checked on a phone walking between jobs, in a couple of seconds.",
 "I am never sitting down when I read these now; it is a glance on a small screen.",
],
"replyformat/compare": [
 "I am running a procurement and have to score five suppliers on the same eight criteria.",
 "The decision now involves weighing several candidates against an identical set of dimensions.",
 "I have to assess a shortlist where every option is judged on exactly the same measures.",
],
"replyformat/argument": [
 "The committee rejected my last submission because the points were asserted without showing how they follow.",
 "I have to present to a panel that will not accept a claim unless the reasoning connecting it is visible.",
 "They sent it back saying the conclusions did not obviously follow from anything.",
],
"replyformat/matrix": [
 "The review now covers forty items, each scored on the same six measures.",
 "I am assessing dozens of entries against an identical checklist.",
 "This has grown into a large set of items all judged on the same criteria.",
],
"replyformat/narrative": [
 "I listen to these through a screen reader on the commute rather than looking at them.",
 "I have moved to audio for this; it gets read aloud to me while I drive.",
 "These come to me spoken now, not on a screen.",
],
"replyformat/collapse": [
 "I work in a narrow terminal now and wide layouts wrap into nonsense.",
 "The window I use is about sixty characters across and anything wider breaks apart.",
 "Since switching to the console, anything laid out in columns is unreadable.",
],

# ---------------------------------------------------------------- music
"music/focus": [
 "I have started long writing sessions and anything with prominent vocals pulls me straight out.",
 "The new work is hours of concentration and words in the background wreck it.",
 "I write all afternoon now and lyrics make it impossible to hold a thought.",
],
"music/ensemble": [
 "I went to a run of small live sessions and got completely absorbed in the spontaneous solos.",
 "A friend took me to hear a quartet several times and the improvised passages were what stayed with me.",
 "Those late sets where nothing is written down have taken over what I listen to.",
],
"music/insomnia": [
 "The clinician told me to keep stimulating audio out of the evening because of the sleep problem.",
 "I have been advised to avoid anything with drive to it after eight, given how badly I sleep.",
 "The sleep clinic was specific about high-intensity sound in the hours before bed.",
],
"music/study": [
 "I work late into the night now and unpredictable rhythmic changes keep breaking my concentration.",
 "Late study sessions mean anything that shifts tempo unexpectedly interrupts me.",
 "The quiet night work does not survive anything with sudden changes in it.",
],
"music/improv": [
 "A friend took me to a series of small sessions where the musicians respond to each other in real time.",
 "I have been going to nights where nothing is arranged and the players react to one another.",
 "Those live rooms where it is all made up between them have got under my skin.",
],
"music/band": [
 "I joined a group in September and rehearsals are learning amplified riff-driven parts every week.",
 "The band I am in works through loud guitar material at every practice.",
 "I play with a group now whose whole repertoire is built on distorted riffs.",
],
"music/gym": [
 "I started a heavy lifting programme and need something with real drive behind the sets.",
 "Training is intense now and I want momentum coming out of the headphones.",
 "The new programme is all-out effort and needs something pushing underneath it.",
],

# ---------------------------------------------------------------- reading
"reading/research": [
 "I have taken on a project involving genetics and I have no background in it at all.",
 "The new work needs me to understand molecular mechanisms I have never studied.",
 "I am on a project where I have to follow technical arguments from scratch.",
],
"reading/archive": [
 "I have been tracing my grandparents and need to understand the decades and places they lived through.",
 "A family project has me trying to make sense of a period I know nothing about.",
 "I am piecing together where my relatives were in the 1930s and what was happening around them.",
],
"reading/accuracy": [
 "Three of the accounts I loved turned out to have invented whole episodes; I have lost trust in the genre.",
 "I keep finding that books I enjoyed contained fabrications, and it has soured the whole thing.",
 "Too many of them have been debunked for me to take any of it at face value now.",
],
"reading/course": [
 "I moved into a research group where everyone discusses new findings and the library is excellent.",
 "The new workplace is full of people recommending technical material, and it is all on the shelves.",
 "Since joining the lab I am surrounded by talk of recent discoveries and easy access to the literature.",
],
"reading/exhaustion": [
 "I spend the whole day on dense technical documents and have nothing left for anything demanding.",
 "After eight hours of analysis my concentration is finished by the evening.",
 "The workload is all heavy analytical material now and I cannot face more of it afterwards.",
],
"reading/centenary": [
 "I have been asked to prepare material marking a hundred years since the flood here.",
 "There is a commemoration next year and I am putting together the background on the period.",
 "The council asked me to write up what happened in the decade before the disaster.",
],

# ---------------------------------------------------------------- travel
"travel/burnout": [
 "After eight months of overtime I cannot face anything that needs planning or decisions.",
 "I am completely spent and want a trip where nothing has to be organised.",
 "The overtime has finished me; I need somewhere I do not have to think.",
],
"travel/photography": [
 "I have started a serious project photographing wildlife and need access to open country.",
 "The photography has turned into something I take seriously, and it is all animals and landscape.",
 "I bought a long lens and now everything is about getting to where the wildlife is.",
],
"travel/crowding": [
 "The last three trips were queues and packed streets from morning to night.",
 "Every recent holiday has been shoulder to shoulder and I came back more tired than I left.",
 "I have had enough of two-hour queues and pavements you cannot move on.",
],
"travel/mobility": [
 "I did my ankle badly and cannot manage long distances over uneven ground for several months.",
 "The ankle injury rules out anything strenuous on rough terrain until at least the autumn.",
 "I have been told no long days on broken ground while the ankle heals.",
],
"travel/architecture": [
 "I started studying architecture and want to see the major works in person.",
 "The course means I need to visit the buildings and collections I am writing about.",
 "I am doing a degree now that requires seeing particular buildings first hand.",
],
"travel/conference": [
 "My field runs its meetings in the big centres and I am at three or four a year now.",
 "The professional events I have to attend are all held in large urban venues.",
 "Work travel now means whichever city is hosting the conference that season.",
],
"travel/hiking": [
 "I committed to a multi-day trek in the spring and need to build up on real ground.",
 "There is a long route I am doing next year and preparation means proper terrain.",
 "I signed up for a week-long walk and have to get the miles in on varied ground.",
],

# ---------------------------------------------------------------- mealprep
"mealprep/schedule": [
 "The new commute takes the evenings but Sunday afternoons are completely free.",
 "Weeknights are gone since the job moved, though I have a clear block at the weekend.",
 "I have no time midweek any more, but one long stretch on Sunday is untouched.",
],
"mealprep/renovation": [
 "A pipe went and I have had no stove or sink for three weeks with no end in sight.",
 "The plumbing failure means the kitchen is unusable until at least the end of the month.",
 "There is no running water or hob in here while the repairs drag on.",
],
"mealprep/savings": [
 "I am saving for a deposit and worked out how much the ordered food alone was costing me.",
 "The savings target means going through where the money goes, and that was the biggest line.",
 "I am putting money aside seriously now and the discretionary spending has to go.",
],
"mealprep/hosting": [
 "The weekly group has grown to twenty people and I am the one bringing the food.",
 "I have to feed a large number every Thursday and I have very little time on the day.",
 "There is a community night now where I provide for everyone, on a work evening.",
],
"mealprep/burn": [
 "There was an oil fire in here last month and I have not been able to face the hob since.",
 "The kitchen fire did real damage and I get anxious now even standing at the cooker.",
 "After the fire I cannot bring myself to use the stove.",
],
"mealprep/freezer": [
 "The chest freezer died and I cannot afford to replace it, so nothing keeps beyond a day.",
 "Losing the freezer means I have no way to hold anything over.",
 "There is no cold storage any more; whatever exists has to be eaten that day.",
],
"mealprep/shifts": [
 "The rota went to rotating weekends and there is no predictable free block any more.",
 "I never know which day is off now, so the long session I relied on has gone.",
 "The shift pattern changed and the one clear afternoon a week disappeared with it.",
],

# ---------------------------------------------------------------- communication
"communication/urgent": [
 "The new role means an hour's delay can stop the line, and decisions need rapid back-and-forth.",
 "Everything is time-critical now; waiting on a reply costs money by the minute.",
 "Issues here have to be resolved immediately or production halts.",
],
"communication/traveling": [
 "I am between sites all day now and only ever have a moment on a handset.",
 "The new routine keeps me away from a desk; I get seconds at a time.",
 "I am rarely at a computer any more, just brief gaps while moving around.",
],
"communication/hearing": [
 "My hearing has changed and I cannot make out compressed audio over a line any more.",
 "I miss half of what is said down a line now; the compression makes it unintelligible.",
 "Speech over a connection has become unreliable for me since the hearing loss.",
],
"communication/record": [
 "Compliance now requires every approval to exist as a searchable document.",
 "The audit process means agreements have to be retrievable later, in writing.",
 "Everything decided has to be logged somewhere it can be found again.",
],
"communication/harassment": [
 "Tone got misread three times in short exchanges and it has damaged two working relationships badly.",
 "Several important conversations went wrong because brevity read as rudeness.",
 "Short written back-and-forth keeps causing conflict I then have to repair.",
],
"communication/contract": [
 "A legal process means every agreement has to be attributable and produced later if asked.",
 "There is litigation running and anything agreed must exist in a retainable form.",
 "The solicitors need everything in a form that can be disclosed.",
],

# ---------------------------------------------------------------- learning
"learning/manual": [
 "The certification is assessed by completing procedures under observation, not by recall.",
 "I have to demonstrate the whole sequence physically in front of an assessor.",
 "They sign you off on performance now, not on what you can describe.",
],
"learning/visual": [
 "I could not work out from the text how the tools are meant to be positioned relative to each other.",
 "Written descriptions of the spatial arrangement kept leaving me stuck.",
 "The instructions describe movements I cannot picture from words alone.",
],
"learning/bandwidth": [
 "We moved and the connection here will not carry anything streamed reliably.",
 "The line at the new place drops constantly; nothing streamed gets through.",
 "There is no usable broadband where we are now.",
],
"learning/equipment": [
 "The workshop closed and I cannot afford to replace what was in it.",
 "I have lost access to the machines and buying my own is out of reach.",
 "The space with the equipment shut down in June and there is nowhere else.",
],
"learning/deadline": [
 "The deadline moved up and I have three evenings to cover a term's worth of material.",
 "There is no time for repeated attempts now; the assessment is next week.",
 "The window collapsed to a few nights and there is a lot to get through.",
],
"learning/apprentice": [
 "I have been placed with a mentor who only signs off work done in front of him.",
 "The new arrangement is supervised execution; nothing counts unless he watches it.",
 "My assessor credits what I do under observation, nothing else.",
],

# ---------------------------------------------------------------- shopping
"shopping/relocation": [
 "We moved and the nearest sizeable retail is a three-hour round trip.",
 "There is nothing within reach here; the closest place of any size is most of a day.",
 "The move put us a long way from anywhere with real choice.",
],
"shopping/freshness": [
 "I have started cooking to whatever is in season and it changes week to week.",
 "The new cooking depends on produce picked within a day or two.",
 "What I make now follows the local harvest, so it has to be very fresh.",
],
"shopping/delivery": [
 "Four expensive orders in a row arrived damaged, late, or not at all.",
 "I have lost hundreds to things that never turned up or came broken.",
 "The last several deliveries all went wrong and I am done chasing refunds.",
],
"shopping/walking": [
 "I did my leg and cannot manage prolonged walking or carrying bags for a while.",
 "The injury rules out being on my feet for long stretches or carrying anything heavy.",
 "I am not able to walk far or carry much until this heals.",
],
"shopping/community": [
 "I went to a neighbourhood event and have got to know several of the producers since.",
 "Since that street fair I have built up relationships with the people who grow it.",
 "I know the growers by name now after getting involved locally.",
],
"shopping/winter": [
 "The open-air pitches shut from November and only the covered places stay open.",
 "Everything outdoors closes for the season; the indoor premises are all that is left.",
 "From next month the outdoor traders are gone until spring.",
],
}
