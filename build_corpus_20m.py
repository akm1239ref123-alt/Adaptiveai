from pathlib import Path
import random

ROOT = Path(__file__).parent
OUT = ROOT / 'data' / 'train_20m.txt'
random.seed(20261004)

names = ['Maya','Omar','Lina','Adam','Nora','Samir','Yara','Leo','Hana','Karim','Mila','Ziad']
places = ['a small town','a busy city','a quiet village','a school lab','a library','a workshop','a park','a computer room','a coastal village','a mountain cabin']
topics = ['Python','HTML','CSS','JavaScript','Linux','databases','networks','algorithms','machine learning','physics','biology','astronomy','mathematics']
concepts = {
    'Python':'a general-purpose programming language with readable syntax',
    'HTML':'a markup language used to structure web content',
    'CSS':'a stylesheet language used to control presentation and layout',
    'JavaScript':'a programming language commonly used for interactive web applications',
    'Linux':'a family of operating systems built around the Linux kernel',
    'databases':'systems for storing and retrieving organized information',
    'networks':'systems that allow computers and devices to communicate',
    'algorithms':'step-by-step procedures for solving problems',
    'machine learning':'a way to learn patterns from data by adjusting model parameters',
    'physics':'the study of matter, energy, motion, and interactions',
    'biology':'the study of living organisms',
    'astronomy':'the study of objects and phenomena beyond Earth',
    'mathematics':'a field concerned with quantities, structures, patterns, and logical relationships',
}

def qa(q,a):
    return f'User: {q}\nPersona: {a}\n\n'

blocks=[]
# Core conversational behavior
core = [
('Hello', 'Hello! I am Persona. What would you like to talk about?'),
('Can you keep your answer short?', 'Yes. I will give the direct answer first and avoid unnecessary detail.'),
('What should you do when you are unsure?', 'I should say that I am unsure and avoid presenting a guess as a fact.'),
('Can you learn from corrections?', 'Yes. A correction can become a training example for a later training run.'),
('Can you explain things simply?', 'Yes. I can start with the main idea, use a small example, and add technical detail only when useful.'),
('What makes a good answer?', 'A good answer is relevant, clear, honest about uncertainty, and matched to the question.'),
('What should you do if a question is ambiguous?', 'If the missing detail changes the answer, I should ask a focused clarification question.'),
]
for q,a in core: blocks.append(qa(q,a))

# Knowledge templates
for topic in topics:
    desc = concepts[topic]
    variants = [
        f'What is {topic}?',
        f'Can you explain {topic}?',
        f'Tell me about {topic}.',
        f'Why is {topic} useful?',
        f'What is a simple way to learn {topic}?',
    ]
    answers = [
        f'{topic.capitalize()} is {desc}.',
        f'In simple terms, {topic} is {desc}. A useful way to understand it is to study one small example at a time.',
        f'{topic.capitalize()} is useful because it helps people solve problems or understand systems in a structured way.',
        f'A practical way to learn {topic} is to study the basic concepts, build small examples, inspect mistakes, and gradually increase difficulty.',
    ]
    for i,q in enumerate(variants):
        blocks.append(qa(q, answers[i % len(answers)]))

# Math examples with varied wording
for a,b in [(2,2),(3,7),(5,9),(12,8),(14,6),(25,4),(18,7),(21,3),(144,12),(81,9)]:
    blocks.append(qa(f'What is {a} plus {b}?', f'{a} + {b} = {a+b}.'))
    blocks.append(qa(f'Calculate {a} times {b}.', f'{a} × {b} = {a*b}.'))

# Coding guidance
coding = [
('How do I debug a program?', 'Reproduce the problem, read the exact error, isolate the smallest failing part, change one thing at a time, and test again.'),
('What should I include in a bug report?', 'Include the expected result, actual result, exact error, relevant code, inputs, environment, and steps to reproduce the problem.'),
('How do I learn programming?', 'Practice by building small projects, reading documentation, debugging your own mistakes, and gradually increasing project complexity.'),
('What does a loop do?', 'A loop repeats instructions for a condition or for each item in a sequence.'),
('What does an if statement do?', 'An if statement chooses whether to run code based on a condition.'),
('What is a function?', 'A function is a reusable block of code that can accept inputs and produce a result or side effect.'),
('What is a variable?', 'A variable is a named place or binding used to refer to a value in a program.'),
('Why use version control?', 'Version control records changes so you can review history, collaborate, and recover earlier versions.'),
]
for q,a in coding: blocks.append(qa(q,a))

# Short multi-turn conversations, randomized combinations
for i in range(60000):
    name = random.choice(names); place = random.choice(places); topic = random.choice(topics)
    task = random.choice(['learn','debug','build','explain','practice'])
    if task == 'learn':
        q1 = f'{name} is at {place}. They want to learn {topic}. What should they do first?'
        a1 = f'Start with the basic concepts of {topic}, then practice with a small task. It is better to understand a few ideas well than to memorize everything at once.'
        q2 = 'What should they do after that?'
        a2 = f'They should build a small example, inspect mistakes, and increase the difficulty gradually.'
    elif task == 'debug':
        q1 = f'{name} is working on {topic} and the project does not behave as expected. What is the first step?'
        a1 = 'First reproduce the problem consistently and record what actually happens. Then compare it with the expected behavior.'
        q2 = 'Should they rewrite the whole project immediately?'
        a2 = 'Usually no. Isolate the smallest failing part first so the cause is easier to identify.'
    elif task == 'build':
        q1 = f'{name} wants to build a small {topic} project at {place}. Is that a good way to learn?'
        a1 = 'Yes. A small project turns abstract ideas into practice and gives you concrete problems to solve.'
        q2 = 'How large should the first project be?'
        a2 = 'Small enough to finish in a short period, but large enough to use several concepts together.'
    elif task == 'explain':
        q1 = f'{name} asks: can you explain {topic} like I am a beginner?'
        a1 = f'Sure. Think of {topic} as {concepts[topic]}. Start with one simple example before studying advanced details.'
        q2 = 'Can you make that even simpler?'
        a2 = 'Yes. Learn the basic idea first, use it in a tiny example, and then build on that example.'
    else:
        q1 = f'{name} wants to practice {topic}. What is a useful exercise?'
        a1 = f'Choose a small problem involving {topic}, solve it without copying a complete solution, then compare your result with reliable documentation.'
        q2 = 'What if the exercise is too hard?'
        a2 = 'Break it into smaller steps and solve the easiest step first.'
    blocks.append(qa(q1,a1)); blocks.append(qa(q2,a2))

# Stories/dialogue: improves natural language transitions and multi-sentence generation.
for i in range(25000):
    name1, name2 = random.sample(names, 2)
    place = random.choice(places)
    topic = random.choice(topics)
    goal = random.choice(['finish a project','understand a new idea','fix a mistake','prepare for a test','build a small tool'])
    blocks.append(
        f'{name1} and {name2} were working in {place}. {name1} wanted to {goal}. '
        f'{name2} suggested starting with a small example about {topic}. '
        f'{name1} tried the example, found a mistake, and checked each step carefully. '
        f'After fixing the mistake, the example worked. They decided to make the next task slightly harder.\n\n'
    )

random.shuffle(blocks)
text=''.join(blocks)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(text, encoding='utf-8')
print(f'Wrote {OUT}: {len(text):,} chars, approx {len(text.encode()):,} bytes, {len(blocks):,} examples')
