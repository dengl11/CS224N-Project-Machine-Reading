import matplotlib.pyplot as plt;
import collections

# Answer Length Analysis.
answers = []
with open('dev.answer') as f:
  for line in f.readlines():
    # remove '\n' at the end.
    answers.append(line[:-1])

# answers_length_count = collections.Counter([len(x.split()) for x in answers])
#
#
# y = answers_length_count.values()
# x = answers_length_count.keys()
# width = 1/1.5
# plt.bar(x, y, width, color="blue")
# plt.ylabel('Count')
# plt.xlabel('Answer Word Length')
# plt.show()


# Context Length Analysis.
context = []
with open('dev.context') as f:
  for line in f.readlines():
    # remove '\n' at the end.
    context.append(line[:-1])

# context_length_count = collections.Counter([len(x.split()) for x in context])
#
# y = context_length_count.values()
# x = context_length_count.keys()
# width = 1/1.5
# plt.bar(x, y, width, color="blue")
# plt.ylabel('Count')
# plt.xlabel('Context Word Length')
# plt.show()


# Analyze Answer in Context Starting Position (in terms of context length percentage, ~X% of context length)
# starting_position_count = collections.Counter()
# for index, answer in enumerate(answers):
#   one_context = context[index]
#   context_word_length = len(one_context.split())
#   previous_context_word_length = len(one_context[:one_context.find(answer)].split())
#   starting_position_count[round((1.0 * previous_context_word_length / context_word_length) * 100, 0)] += 1
#
# y = [starting_position_count[position] for position in starting_position_count.keys()]
# x = starting_position_count.keys()
# width = 1/1.5
# plt.bar(x, y, width, color="blue")
# plt.ylabel('Count')
# plt.xlabel('Answer Starting Position (~X% context word length)')
# plt.show()


# # Analyze Number of Words before Answer in Context
# numer_of_words_before_answer_count = collections.Counter()
# for index, answer in enumerate(answers):
#   one_context = context[index]
#   context_word_length = len(one_context.split())
#   previous_context_word_length = len(one_context[:one_context.find(answer)].split())
#   numer_of_words_before_answer_count[previous_context_word_length] += 1
#
# y = [numer_of_words_before_answer_count[position] for position in numer_of_words_before_answer_count.keys()]
# x = numer_of_words_before_answer_count.keys()
# width = 1/1.5
# plt.bar(x, y, width, color="blue")
# plt.ylabel('Count')
# plt.xlabel('Number of Word Before Answer in Context')
# plt.show()
