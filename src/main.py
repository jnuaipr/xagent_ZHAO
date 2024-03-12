from mathematical_processing import normalize_sum
from mathematical_processing import TF_Google_Weighted
from mathematical_processing import number_to_text
from text_analysis import finance_text_analysis, jurisprudence_text_analysis, medical_text_analysis, TF_IDF_jurisprudence,TF_IDF_medical,TF_IDF_finance

text = "Recent advances in cardiology have led to new therapies for heart diseases antibiotics immunosuppressants Political Action Committee Plea Bargain,"



# 计算在谷歌发布的模型的隶属度
membership_score1_1 = medical_text_analysis.calculate_domain_affinity(text)
membership_score2_1 = jurisprudence_text_analysis.calculate_domain_affinity(text)
membership_score3_1 = finance_text_analysis.calculate_domain_affinity(text)


# 计算在TF_IDF计算下的隶属度
membership_score1_2 = TF_IDF_medical.medical_membership_score_normalized(text)
membership_score2_2 = TF_IDF_jurisprudence.legal_membership_score_normalized(text)
membership_score3_2 = TF_IDF_finance.finance_membership_score_normalized(text)

# 融合两种隶属度方式
membership_score1 = TF_Google_Weighted.TF_Google_Weighted_func(membership_score1_1, membership_score1_2)
membership_score2 = TF_Google_Weighted.TF_Google_Weighted_func(membership_score2_1, membership_score2_2)
membership_score3 = TF_Google_Weighted.TF_Google_Weighted_func(membership_score3_1, membership_score3_2)
print("隶属度于医学的分数:", membership_score1_1,membership_score1_2)
print("隶属度于医学的分数:", membership_score2_1,membership_score2_2)
print("隶属度于医学的分数:", membership_score3_1,membership_score3_2)


# 归一化隶属度
normalized_values = normalize_sum.normalize_sum_to_one(membership_score1, membership_score2, membership_score3)
membership_score1 = normalized_values[0]
membership_score2 = normalized_values[1]
membership_score3 = normalized_values[2]
membership = number_to_text.plot_gaussians_with_point_and_return_membership(membership_score1)  # 输入x值
print("在医学领域的隶属度是", membership)
print("隶属度于医学的分数:", membership_score1)
print("隶属度于法学的分数:", membership_score2)
print("隶属度于金融学的分数:", membership_score3)


'''
print("调用本地医学小模型")
results = gpt2_generation.generate_text(text)
# 打印结果
print('\nInput:\n' + 100 * '-')
print('\033[96m' + text + '\033[0m')
print('\nOutput:\n' + 100 * '-')
for result in results:
    print('\033[92m' + result + '\033[0m\n')
# 如果 membership_score2 是最大的，执行操作2
print("调用本地法学小模型")
print('\nInput:\n' + 100 * '-')
print('\033[96m' + text + '\033[0m')
print('\nOutput:\n' + 100 * '-')
results = legal_generation.generate_text(text)
for i, result in enumerate(results):
    print(f"{i + 1}:\n{result}\n")
# 如果 membership_score3 是最大的，执行操作3
print("调用本地金融小模型")
# 打印生成的文本
results = finance_generation.generate_text(text)
print('\nInput:\n' + 100 * '-')
print('\033[96m' + text + '\033[0m')
print('\nOutput:\n' + 100 * '-')
for i, result in enumerate(results):
    print(f"Generated Text {i + 1}:\n{result}\n")

GPT_api_request.send_request(text)


if:调用大模型
      执行脱敏：调用Desensitition;
'''



