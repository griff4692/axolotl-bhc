INSTRUCTIONS = {
    'baseline': 'Generate the BRIEF HOSPITAL COURSE summary.',
    'frost': 'Generate the BRIEF HOSPITAL COURSE summary using only the medical entities (PROBLEMS, TREATMENTS, and TESTS) provided.',
    'sent_frost': 'Generate the next sentence of the BRIEF HOSPITAL COURSE summary using only the medical entities (PROBLEMS, TREATMENTS, and TESTS) listed below.',
    'sent_focus': 'Play close attention to entities in double brackets {{ }} when generating the next sentence of the BRIEF HOSPITAL COURSE summary.',
    'focus': 'Play close attention to entities in double brackets {{ }} when generating the BRIEF HOSPITAL COURSE summary.',
    'sent_planning': 'Retrieve a subset of the medical entities in double brackets {{ }} and use them to generate the next sentence of the BRIEF HOSPITAL COURSE summary.',
    'sent_planning_with_reuse': 'Retrieve a subset of the medical entities in double brackets ({{ }} or << >>) and use them to generate the next sentence of the BRIEF HOSPITAL COURSE summary.',
    'debug': 'Retrieve the {{ }} bracketed sentence.',
    'neg_plan': 'Pay NO attention to entities in double brackets {{ }} when generating the next sentence of the BRIEF HOSPITAL COURSE summary.'
}
