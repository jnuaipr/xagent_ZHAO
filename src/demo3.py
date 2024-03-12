def get_nth_word(sentence, n):
    words = sentence.split()

    if 0 <= n < len(words):
        return words[n]
    else:
        return f"Word at position {n} is out of range for the given sentence."

# 示例句子
input_sentence = "This is a sample sentence for demonstration purposes."

# 要查找的单词位置
word_position_to_find = 3

# 查找第N个单词
result = get_nth_word(input_sentence, word_position_to_find)

print(f"The word at position {word_position_to_find} is: {result}")
