"""Offline audit of the supplied LRAT pilot. No inference or source-file edits."""

import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

EXPERIMENT = Path('runs/infoseek-20-baseline-v1')
JUDGE_RUN = 'judge-localcomm-job-59491'
INDEX_RUN = Path('indexes/wiki-25-fa2-cacheoff-b512')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(data):
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def question_number(qid):
    return int(qid.rsplit('_', 1)[-1])


def official_verdict(text):
    """Exactly the original evaluator's regular expression, including its scope."""
    match = re.search(r'correct:\s*(yes|no)', text or '', re.IGNORECASE)
    return match.group(1).lower() if match else None


def json_fallback_verdict(text):
    """Recover an unambiguous JSON verdict only if the original regex fails."""
    original = official_verdict(text)
    if original is not None:
        return original, 'original_regex'
    decoder = json.JSONDecoder()
    verdicts = set()
    for position, character in enumerate(text or ''):
        if character != '{':
            continue
        try:
            obj, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get('correct'), str):
            verdict = obj['correct'].strip().lower()
            if verdict in ('yes', 'no'):
                verdicts.add(verdict)
    if len(verdicts) == 1:
        return verdicts.pop(), 'json_fallback'
    return None, 'unresolved'


def extract_answer(output, termination, error=None, retrieval_errors=None):
    """Mirror assess_answer from the archived runner (no finish_reason supplied).

    The exported summaries do not include per-request finish reasons. Existing
    summaries remain authoritative for any additional length-limit annotation.
    """
    text = re.sub(r'<think>.*?</think>', '', output or '', flags=re.DOTALL).strip()
    if error:
        status = 'request_error'
    elif retrieval_errors:
        status = 'retrieval_error'
    elif '<think>' in text:
        status = 'unfinished_thinking'
    else:
        if '</think>' in text:
            text = text.rsplit('</think>', 1)[1].strip()
        if '<tool_call>' in text or '</tool_call>' in text:
            status = 'tool_call_instead_of_answer'
        elif '<answer>' in text and '</answer>' not in text:
            status = 'unfinished_answer'
        else:
            match = re.search(r'<answer>(.*?)</answer>', text, flags=re.DOTALL)
            if match:
                text = match.group(1).strip()
            if not text:
                status = 'empty_answer'
            elif termination not in ('answer', 'generate an answer as token limit reached'):
                status = 'unexpected_termination'
            else:
                status = 'answer_text_present'
    completed = status == 'answer_text_present'
    return {'final_answer': text if completed else '', 'answer_status': status,
            'completed_with_answer': completed}


def judge_template(path):
    """Read the template as an AST literal; never import the GPU evaluator."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'JUDGE_TEMPLATE' for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == 'strip':
                return ast.literal_eval(value.func.value).strip()
            return ast.literal_eval(value)
    raise ValueError('JUDGE_TEMPLATE not found')


def audit_project(project_root):
    root = Path(project_root).resolve()
    folder = root / EXPERIMENT / JUDGE_RUN
    protocol = load_json(folder / 'judge-protocol.json')
    recorded = load_json(folder / 'comparison.json')
    checks = []

    def check(condition, label):
        if not condition:
            raise ValueError('Integrity check failed: ' + label)
        checks.append(label)

    check(sha256(folder / 'official-evaluate.py') == protocol['evaluator_sha256'], 'Original evaluator SHA-256')
    check(sha256(folder / 'evaluate-local-comm.py') == protocol['executed_evaluator_sha256'], 'Executed evaluator SHA-256')
    check(sha256(folder / 'ground-truth.tsv') == protocol['ground_truth_sha256'], 'Ground-truth SHA-256')
    with (folder / 'ground-truth.tsv').open(encoding='utf-8', newline='') as stream:
        gt_rows = list(csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_NONE))
    truth = {row[0].strip(): {'question': row[1].strip(), 'answer': row[2].strip()} for row in gt_rows}
    selected = protocol['selected_questions']
    check(len(truth) == 300, '300 distinct official ground-truth questions')
    check(selected == [[row[0].strip(), row[1].strip()] for row in gt_rows[:20]], 'Pilot is the first 20 official questions')
    check(len(protocol['inputs']) == 40, 'Manifest has 40 frozen inputs')
    check({p.name for p in (folder / 'inputs').glob('*.json')} == set(protocol['inputs']), 'Frozen input filenames match manifest')
    check(load_json(folder / 'communication-test.json')['passed'] is True, 'Two-GPU communication pretest passed')

    indexes = {}
    summaries = {}
    conditions = {}
    rows = []
    cases = {}
    expected_ids = {qid for qid, _ in selected}
    all_results = load_json(folder / 'official-all-results.json')
    all_by_file = {row['file']: row for row in all_results['details']}
    check(len(all_by_file) == 40, 'Official aggregate has 40 distinct outcomes')
    template = judge_template(folder / 'official-evaluate.py')
    check(template == judge_template(folder / 'evaluate-local-comm.py'), 'Judge prompt unchanged in private evaluator copy')

    for name in ('qwen', 'lrat'):
        identity_record = protocol['protocols'][name]
        identity = identity_record['identity']
        check(canonical_digest(identity) == identity_record['identity_sha256'], name + ': agent protocol identity hash')
        check(identity['queries'] == selected, name + ': identical pilot questions')
        check(sha256(root / 'run_infoseek_sample.py') == identity['sample_runner_sha256'], name + ': archived sample-runner hash')
        check(sha256(root / 'run_infoseek_pilot.py') == identity['pilot_helper_sha256'], name + ': archived pilot-helper hash')
        index_path = root / INDEX_RUN / name / 'metadata.json'
        check(sha256(index_path) == identity['index_metadata_sha256'], name + ': index metadata hash')
        indexes[name] = load_json(index_path)
        check(indexes[name]['build_identity']['encoding_run']['corpus_sha256'] == identity['corpus_sha256'], name + ': index/agent corpus hash agreement')
        check(sum(x['rows'] for x in indexes[name]['build_identity']['inputs']) == indexes[name]['documents'], name + ': shard row count agreement')
        summary = load_json(root / EXPERIMENT / name / 'sample-summary.json')
        summaries[name] = summary
        records = {item['query_id']: item for item in summary['results']}
        check(set(records) == expected_ids and summary['processed'] == 20, name + ': all generation outcomes retained')
        check(summary['infrastructure_errors'] == 0, name + ': no reported generation infrastructure errors')
        result = load_json(folder / (name + '-eval.json'))
        check({item['query_id'] for item in result['details']} == expected_ids and len(result['details']) == 20, name + ': 20 judge outcomes')

        for item in sorted(result['details'], key=lambda x: question_number(x['query_id'])):
            qid = item['query_id']
            source = folder / 'inputs' / item['file']
            check(sha256(source) == protocol['inputs'][item['file']]['sha256'], name + '/' + qid + ': frozen trajectory hash')
            check(item == all_by_file[item['file']], name + '/' + qid + ': split and aggregate verdicts agree')
            run = load_json(source)
            record = records[qid]
            output = (run['result'][-1].get('output', '') or '') if run.get('result') else ''
            assessment = extract_answer(output, record['termination'], record['error'], record['retrieval_errors'])
            check(assessment['answer_status'] == record['answer_status'] and assessment['final_answer'] == record['final_answer'] and assessment['completed_with_answer'] == record['completed_with_answer'], name + '/' + qid + ': output extraction agrees with generation summary')
            counts = run.get('tool_call_counts', {}) or {}
            steps = sum(counts.values())
            check(steps == item['steps'] == protocol['inputs'][item['file']]['steps'], name + '/' + qid + ': tool-call counts agree')
            check(counts == record['tool_calls'], name + '/' + qid + ': summary and trajectory tool counts agree')
            old_verdict = official_verdict(item['judge_reason'])
            new_verdict, parse_source = json_fallback_verdict(item['judge_reason'])
            # Preserve the released evaluator's independent unfinished-output guard.
            rejected = output is None or 'tool_call' in output or 'user_query' in output
            official_correct = old_verdict == 'yes' and not rejected
            check(bool(item['correct']) == official_correct, name + '/' + qid + ': reproduce original correctness calculation')
            format_aware = new_verdict == 'yes' and not rejected
            completed = record['completed_with_answer']
            prompt = template.format(question=truth[qid]['question'], response=output,
                                     correct_answer=truth[qid]['answer'])
            row = {
                'retriever': name, 'query_id': qid, 'official_correct': bool(item['correct']),
                'regex_verdict': old_verdict, 'format_aware_verdict': new_verdict,
                'verdict_source': parse_source, 'official_output_rejected': rejected,
                'format_aware_same_guard_correct': format_aware,
                'answer_status': record['answer_status'], 'completed_with_answer': completed,
                'final_answer_gate_diagnostic': bool(item['correct']) and completed,
                'credited_without_final_answer': bool(item['correct']) and not completed,
                'steps': int(steps), 'search_calls': counts.get('search', 0),
                'get_document_calls': counts.get('get_document', 0),
                'generation_seconds': record['seconds'], 'termination': record['termination'],
                'agent_raw_output_chars': len(output), 'final_answer_chars': len(record['final_answer']),
                'judge_output_chars': len(item['judge_reason']),
                'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
            }
            rows.append(row)
            cases[(name, qid)] = {'record': record, 'raw': run, 'judge': item,
                                 'question': truth[qid]['question'], 'reference_answer': truth[qid]['answer'],
                                 'raw_output': output, 'clean_final_answer': record['final_answer'],
                                 'prompt': prompt}

        group = [row for row in rows if row['retriever'] == name]
        metrics = {
            'n': len(group), 'official_correct': sum(x['official_correct'] for x in group),
            'format_aware_same_guard_correct': sum(x['format_aware_same_guard_correct'] for x in group),
            'completed_final_answers': sum(x['completed_with_answer'] for x in group),
            'credited_without_final_answer': sum(x['credited_without_final_answer'] for x in group),
            'final_answer_gate_diagnostic_correct': sum(x['final_answer_gate_diagnostic'] for x in group),
            'unparsed_regex_verdicts': sum(x['regex_verdict'] is None for x in group),
            'unresolved_after_json_fallback': sum(x['format_aware_verdict'] is None for x in group),
            'avg_steps': mean(x['steps'] for x in group),
            'at_50_tool_calls': sum(x['steps'] == 50 for x in group),
            'answer_status_counts': dict(Counter(x['answer_status'] for x in group)),
        }
        check(metrics['official_correct'] == recorded['conditions'][name]['correct'], name + ': recorded success count reproduced')
        check(abs(metrics['avg_steps'] - recorded['conditions'][name]['metrics']['Avg Steps']) < 1e-9, name + ': recorded average steps reproduced')
        conditions[name] = metrics

    qi = protocol['protocols']['qwen']['identity']
    li = protocol['protocols']['lrat']['identity']
    for key in ('protocol', 'queries', 'corpus_sha256', 'source_hashes', 'server_model',
                'server_context_limit', 'agent_config_sha256', 'agent_generation_config', 'agent_weight_files'):
        check(qi[key] == li[key], 'Shared condition: ' + key)
    check(indexes['qwen']['documents'] == indexes['lrat']['documents'], 'Same indexed passage count')
    check(indexes['qwen']['files']['docids.pkl']['sha256'] == indexes['lrat']['files']['docids.pkl']['sha256'], 'Same document-ID mapping hash')

    report = {
        'schema_version': 1, 'source_judge_run': str(EXPERIMENT / JUDGE_RUN),
        'original_files_modified': False, 'inference_performed': False,
        'checks_passed': len(checks), 'conditions': conditions,
        'format_recoveries': [x for x in rows if x['verdict_source'] == 'json_fallback'],
        'credited_without_final_answer': [x for x in rows if x['credited_without_final_answer']],
        'interpretation': {
            'recorded_scores': 'Reproduced original evaluator outputs; not validated final-answer accuracy.',
            'format_audit': 'JSON fallback retains the original independent tool-call/user-query rejection guard.',
            'gate_diagnostic': 'Existing judge labels AND final-answer presence. No clean-answer rejudging was performed; not a replacement accuracy estimate.',
            'next_evaluation': 'Judge cleaned completed final answers only; mark nonanswers failed; retain all 20 questions per condition.',
        },
    }
    return {'root': root, 'folder': folder, 'protocol': protocol, 'recorded': recorded,
            'indexes': indexes, 'summaries': summaries, 'checks': checks, 'rows': rows,
            'cases': cases, 'report': report}


def write_exports(audit, destination):
    """Write derived audit artifacts only. Original experiment files stay intact."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'pilot-audit.json').write_text(json.dumps(audit['report'], indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (destination / 'per-question-audit.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit['rows'][0]))
        writer.writeheader()
        writer.writerows(audit['rows'])
    rejudge = []
    for (name, qid), case in sorted(audit['cases'].items(), key=lambda x: (x[0][0], question_number(x[0][1]))):
        record = case['record']
        rejudge.append({
            'retriever': name, 'query_id': qid, 'question': case['question'],
            'reference_answer': case['reference_answer'], 'response': case['clean_final_answer'],
            'requires_judge': record['completed_with_answer'],
            'automatic_failure_reason': None if record['completed_with_answer'] else record['answer_status'],
            'source_file': case['judge']['file'],
        })
    with (destination / 'clean-answer-rejudge-inputs.jsonl').open('w', encoding='utf-8') as stream:
        for record in rejudge:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    return destination


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    result = audit_project(root)
    target = write_exports(result, root / 'analysis/exports')
    print(json.dumps(result['report'], indent=2, ensure_ascii=False))
    print('Derived files:', target)
