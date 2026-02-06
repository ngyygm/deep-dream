"""
LLM客户端：封装LLM调用，实现三个核心任务
"""
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import uuid

from .models import MemoryCache, Entity


class LLMClient:
    """LLM客户端 - 负责调用大模型完成三个任务"""
    
    # ============ 共享的实体对齐判断原则 ============
    # 用于 judge_entity_match 和 analyze_entity_duplicates 等实体对齐相关函数
    ENTITY_ALIGNMENT_CORE_PRINCIPLES = """
**核心判断方法（三步法）**：
1. 先看名称：名称是否相同、相似、或是别名关系？
2. 再看类型：两个实体描述的是否是同一类型的对象（人物/概念/作品/地点等）？
3. 最后确认：content描述的主体是否是同一个具体对象？

**应该匹配的情况**：
- ✅ 名称相同 + 同一对象的新信息：如已有"刘慈欣（科幻作家）"，新抽取"刘慈欣（获得雨果奖）"→ 同一人的补充信息
- ✅ 别名关系：如"刘慈欣"和"大刘"（同一作家的不同称呼）
- ✅ 格式变体：如"三体"和"《三体》"（同一部小说）
- ✅ 简繁体：如"张三"和"張三"

**绝对不能匹配的情况**：
- ❌ 类型不同：人物 vs 概念/领域（如"刘慈欣"vs"中国科幻"，虽然相关但一个是人一个是领域）
- ❌ 类型不同：人物 vs 作品（如"刘慈欣"vs"《三体》"）
- ❌ 类型不同：具体对象 vs 抽象概念（如"北京"vs"中国"）
- ❌ 仅因content中互相提及就判断为同一实体（相关≠同一）

**关键区分**：
- "同一对象的新信息"→ 匹配（如：同一个人的不同事迹）
- "相关但不同的对象"→ 不匹配（如：作家和他的作品是两个不同实体）
"""

    # ============ 共享的内容合并规则 ============
    # 用于 merge_entity_content 和 merge_relation_content 等内容合并函数
    CONTENT_MERGE_REQUIREMENTS = """
1. **不是简单合并**：不要只是把两个版本的内容拼接在一起，而是要基于两个版本的信息，重新组织和总结
2. **综合信息**：整合两个版本中的所有有效信息，包括旧版本中仍然有效的信息和新版本中的新信息
3. **重新组织**：用新的语言重新组织这些信息，生成一个连贯、完整、准确的描述
4. **去除冗余**：去除重复和冗余的信息，确保描述简洁而完整
5. **保持准确性**：确保总结后的内容准确反映两个版本中的所有重要信息
"""

    # ============ 共享的关系抽取规则 ============
    # 用于 _extract_relations_for_pairs 和 _extract_relations_for_pairs_with_existing 等关系抽取函数
    RELATION_EXTRACTION_CORE_RULES = """
关系应该是概念性的、有意义的描述，而不是简单的标签。
关系是无向的，不区分方向，只表示两个实体之间的关联。

每个关系需要包含：
- entity1_name: 第一个实体名称
- entity2_name: 第二个实体名称
- content: 关系的自然语言描述，专注于描述两个实体之间的具体关系、详细信息以及发生条件

注意：entity1_name和entity2_name只是用来标识关系涉及的两个实体，没有方向性。关系是无向的。

content要求：
1. 专注于描述两个实体之间的具体关系，包括关系的类型、性质、特点等
2. 包含关系的详细信息，如关系的内容、特征、表现等
3. 包含关系的发生条件，如时间、地点、情境等（如果文本中有相关信息）
4. 描述应该简洁明了，不宜过长，重点突出关系的核心信息
"""

    # ============ 共享的关系有效性判断标准 ============
    # 用于 judge_relation_need_create 和 analyze_entity_duplicates 等需要判断关系有效性的函数
    RELATION_VALIDITY_CRITERIA = """
**关联必须是明确的、直接的、有意义的**：
- ✅ 使用关系：一个实体使用另一个实体
- ✅ 包含关系：一个实体包含另一个实体
- ✅ 交互关系：两个实体之间有实际的交互或互动
- ✅ 从属关系：一个实体从属于另一个实体
- ✅ 创建关系：一个实体创建了另一个实体
- ✅ 影响关系：一个实体影响了另一个实体

**不能是模糊的、间接的或牵强的关联**：
- ❌ 只是因为在同一场景中出现，但没有实际的交互或联系
- ❌ 只是因为在同一文本中被提及，但没有明确的关系
- ❌ 只是概念上的相似，但没有实际的关联
- ❌ 只是时间或空间上的接近，但没有实际的关联

**严格标准**：如果不确定两个实体之间是否确实存在明确的、有意义的关联，宁可不创建关系边
"""
    
    @staticmethod
    def _normalize_entity_pair(entity1: str, entity2: str) -> tuple:
        """
        标准化实体对，将实体对按字母顺序排序，使关系无向化
        
        Args:
            entity1: 第一个实体名称
            entity2: 第二个实体名称
        
        Returns:
            标准化后的实体对元组（按字母顺序排序）
        """
        entity1 = entity1.strip()
        entity2 = entity2.strip()
        # 按字母顺序排序，确保(A,B)和(B,A)被视为同一个关系
        if entity1 <= entity2:
            return (entity1, entity2)
        else:
            return (entity2, entity1)
    
    @staticmethod
    def _normalize_entity_name_to_original(entity_name: str, valid_entity_names: set) -> str:
        """
        将LLM返回的实体名称规范化为原始的完整名称（仅精确匹配）
        
        Args:
            entity_name: LLM返回的实体名称
            valid_entity_names: 有效的原始实体名称集合
        
        Returns:
            如果找到精确匹配则返回规范化后的实体名称，否则返回原名称
        """
        entity_name = entity_name.strip()
        
        # 精确匹配：如果已经是完整名称，直接返回
        if entity_name in valid_entity_names:
            return entity_name
        
        # 没有找到精确匹配，返回原名称
        return entity_name
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4", base_url: Optional[str] = None,
                 content_snippet_length: int = 50):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥
            model_name: 模型名称
            base_url: API基础URL（可选，用于自定义API端点）
            content_snippet_length: 传入LLM prompt的实体content最大长度（默认50字符）
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.content_snippet_length = content_snippet_length
        
        # 这里可以根据实际情况选择不同的LLM客户端
        # 支持新旧版本的 OpenAI API
        self.client = None
        self.use_legacy_api = False  # 标记是否使用旧版本 API
        self.openai_module = None
        self._legacy_base_url = None  # 旧版本 API 的 base_url
        
        try:
            import openai
            self.openai_module = openai
            
            if api_key:
                # 检查 openai 版本
                if hasattr(openai, 'OpenAI'):
                    # 新版本 (>= 1.0.0) - 使用 OpenAI 类
                    try:
                        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
                        self.use_legacy_api = False
                    except Exception as e:
                        print(f"警告：初始化新版本 OpenAI 客户端失败: {e}")
                        self.client = None
                else:
                    # 旧版本 (< 1.0.0) - 使用旧 API
                    try:
                        # 设置旧版本 API 的配置
                        openai.api_key = api_key
                        if base_url:
                            # 旧版本可能使用 api_base 或 organization
                            if hasattr(openai, 'api_base'):
                                openai.api_base = base_url
                            # 保存 base_url 以便在调用时使用
                            self._legacy_base_url = base_url
                        else:
                            self._legacy_base_url = None
                        self.use_legacy_api = True
                        self.client = openai  # 使用模块本身作为客户端
                        version = getattr(openai, '__version__', 'unknown')
                        print(f"使用旧版本 openai API (版本: {version})")
                    except Exception as e:
                        print(f"警告：初始化旧版本 OpenAI API 失败: {e}")
                        self.client = None
            else:
                print("提示：未提供 API key，将使用模拟响应模式")
        except ImportError:
            print("警告：未安装openai库，将使用模拟响应模式")
            print("      要使用真实LLM，请安装: pip install openai")
    
    def _is_valid_utf8(self, text: str) -> bool:
        """
        检测文本是否是有效的UTF-8编码
        
        Args:
            text: 待检测的文本
        
        Returns:
            True表示是有效的UTF-8编码，False表示不是
        """
        if not text:
            return True  # 空文本视为有效
        
        try:
            # 尝试将字符串编码为UTF-8字节，然后再解码回来
            # 如果文本包含无效的Unicode字符，这个过程会失败或产生替换字符
            encoded = text.encode('utf-8')
            decoded = encoded.decode('utf-8')
            
            # 检查解码后的文本是否与原始文本相同
            # 如果不同，说明编码/解码过程中出现了问题
            if decoded != text:
                return False
            
            # 检查是否包含Unicode替换字符（\ufffd），这是编码错误的标志
            if '\ufffd' in text:
                return False
            
            return True
        except (UnicodeEncodeError, UnicodeDecodeError):
            # 编码或解码失败，说明不是有效的UTF-8
            return False
        except Exception:
            # 其他异常，保守起见返回True（避免误判）
            return True
    
    def _is_garbled_text(self, text: str) -> bool:
        """
        检测文本是否包含乱码
        
        只检测明确的乱码特征：
        - 包含 \\x 转义序列（如 \\x00, \\xff 等）
        - 包含 \\ufffd Unicode替换字符
        
        Args:
            text: 待检测的文本
        
        Returns:
            True表示包含乱码，False表示正常
        """
        if not text or len(text.strip()) == 0:
            return False
        
        import re
        
        # 1. 检测 \ufffd Unicode替换字符（编码错误的明确标志）
        if '\ufffd' in text:
            return True
        
        # 2. 检测 \x 转义序列（在字符串表示中）
        # 检查文本中是否包含字面的 "\x" 后跟十六进制数字
        if re.search(r'\\x[0-9a-fA-F]{2}', text):
            return True
        
        # 3. 检测实际的 \x 控制字符（在字符串内容中）
        # 检查是否包含 \x00-\x1f 和 \x7f-\xff 范围内的控制字符
        # 使用原始字符串避免转义问题
        if re.search(r'[\x00-\x1f\x7f-\xff]', text):
            # 但排除常见的空白字符（\n, \r, \t）
            control_chars = re.findall(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', text)
            if len(control_chars) > 0:
                return True
        
        return False
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None, max_retries: int = 3) -> str:
        """
        调用LLM的通用方法（带乱码检测和重试机制）
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            max_retries: 最大重试次数（默认3次）
        
        Returns:
            LLM的响应文本
        """
        if self.client is None:
            # 如果没有配置LLM客户端，返回示例响应（用于测试）
            return self._mock_llm_response(prompt)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        last_error = None
        for attempt in range(max_retries):
            try:
                if self.use_legacy_api:
                    # 旧版本 API (0.28.x)
                    import openai
                    # 构建调用参数
                    call_params = {
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": 0.3
                    }
                    # 如果设置了 base_url，可能需要通过其他方式传递
                    # 旧版本可能不支持直接传递 base_url，需要在初始化时设置
                    response = openai.ChatCompletion.create(**call_params)
                    response_text = response.choices[0].message.content
                else:
                    # 新版本 API (>= 1.0.0)
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.3
                    )
                    response_text = response.choices[0].message.content
                
                # 检测是否是有效的UTF-8编码
                if not self._is_valid_utf8(response_text):
                    if attempt < max_retries - 1:
                        print(f"检测到非UTF-8编码的文本，正在重新生成（第 {attempt + 1}/{max_retries} 次尝试）...")
                        print(f"问题内容预览: {response_text[:100]}...")
                        continue
                    else:
                        print(f"警告：检测到非UTF-8编码但已达到最大重试次数，返回原始响应")
                        print(f"问题内容预览: {response_text[:100]}...")
                
                # 检测是否包含乱码
                if self._is_garbled_text(response_text):
                    if attempt < max_retries - 1:
                        print(f"检测到乱码，正在重试（第 {attempt + 1}/{max_retries} 次尝试）...")
                        print(f"乱码内容预览: {response_text[:100]}...")
                        continue
                    else:
                        print(f"警告：检测到乱码但已达到最大重试次数，返回原始响应")
                        print(f"乱码内容预览: {response_text[:100]}...")
                
                # 如果编码有效且没有乱码，或已达到最大重试次数，返回响应
                return response_text
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"LLM调用错误（第 {attempt + 1}/{max_retries} 次尝试）: {e}")
                    print(f"正在重试...")
                    continue
                else:
                    print(f"LLM调用错误（已达最大重试次数）: {e}")
                    return self._mock_llm_response(prompt)
        
        # 如果所有重试都失败，返回模拟响应
        if last_error:
            print(f"所有重试都失败，使用模拟响应")
        return self._mock_llm_response(prompt)
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        清理JSON字符串，修复常见错误
        
        Args:
            json_str: 原始JSON字符串
            
        Returns:
            清理后的JSON字符串
        """
        import re
        # 移除BOM标记
        json_str = json_str.lstrip('\ufeff')
        # 移除首尾空白
        json_str = json_str.strip()
        # 修复中文标点符号
        json_str = json_str.replace('：', ':')  # 中文冒号 -> 英文冒号
        json_str = json_str.replace('，', ',')  # 中文逗号 -> 英文逗号
        json_str = json_str.replace('；', ';')  # 中文分号 -> 英文分号
        json_str = json_str.replace('"', '"')  # 中文左引号 -> 英文引号
        json_str = json_str.replace('"', '"')  # 中文右引号 -> 英文引号
        # 移除可能的尾随逗号（在数组或对象的最后一个元素后）
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        return json_str
    
    def _fix_json_errors(self, json_str: str) -> str:
        """
        尝试修复JSON错误
        
        Args:
            json_str: 有错误的JSON字符串
            
        Returns:
            修复后的JSON字符串
        """
        import re
        # 先进行基本清理
        json_str = self._clean_json_string(json_str)
        
        # 尝试修复常见的JSON错误
        # 1. 修复未转义的换行符（但只在字符串值内部，不在已经转义的地方）
        # 注意：这里需要小心，不要破坏已经正确转义的字符
        
        # 2. 修复无效的Unicode转义序列
        # JSON标准要求 \u 后面必须跟着恰好4个十六进制字符
        # 使用正则表达式更健壮地修复无效的Unicode转义序列
        def fix_unicode_escapes_regex(text):
            """使用正则表达式修复无效的Unicode转义序列"""
            import re
            # 匹配 \u 后面跟着0-3个十六进制字符，然后跟着非十六进制字符或字符串结尾的情况
            # 这包括：
            # - \u 后面直接跟着非十六进制字符（如 \uX, \uXX, \uXXX）
            # - \u 后面跟着0-3个十六进制字符，然后跟着非十六进制字符
            pattern = r'\\u([0-9a-fA-F]{0,3})(?![0-9a-fA-F])'
            
            def replace_invalid_escape(match):
                hex_part = match.group(1)
                if len(hex_part) == 0:
                    # 没有十六进制字符（如 \uX），替换为空格 \u0020
                    return '\\u0020'
                elif len(hex_part) < 4:
                    # 不足4位，用0补齐到4位
                    return '\\u' + hex_part.ljust(4, '0')
                else:
                    # 已经有4位，不应该匹配这个模式，但为了安全起见保持不变
                    return match.group(0)
            
            return re.sub(pattern, replace_invalid_escape, text)
        
        # 使用正则表达式方法修复无效的Unicode转义序列
        json_str = fix_unicode_escapes_regex(json_str)
        
        # 3. 修复未转义的换行符、回车符、制表符（在字符串值中）
        # 注意：这需要在字符串值内部进行，但要避免破坏已经转义的字符
        # 由于这比较复杂，我们只在明显需要的地方进行修复
        
        return json_str
    
    def _clean_markdown_code_blocks(self, text: str) -> str:
        """
        清理文本中的 markdown 代码块标识符
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本（移除 ```markdown 和 ``` 等代码块标识符）
        """
        import re
        # 移除开头的 ```markdown 或 ``` 标识符
        text = re.sub(r'^```\s*markdown\s*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^```\s*\n?', '', text, flags=re.MULTILINE)
        # 移除结尾的 ``` 标识符
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
        # 移除首尾空白
        text = text.strip()
        return text
    
    def _mock_llm_response(self, prompt: str) -> str:
        """模拟LLM响应（用于测试）"""
        prompt_lower = prompt.lower()
        if "更新记忆缓存" in prompt or "memory_cache" in prompt_lower or "创建初始的记忆缓存" in prompt:
            return """当前摘要：正在处理文档内容。当前阅读的是文档的开头部分，介绍了故事的基本背景和主要人物。重要细节包括主要人物的基本信息和故事的初始情境。

自我思考：
- 应该关注：主要人物的身份、性格特点、故事发生的背景环境
- 预判重点：后续情节可能围绕这些主要人物展开，需要留意人物之间的关系和故事的发展方向
- 疑虑：暂无特别疑虑，需要继续阅读以了解故事的发展

系统状态：
- 已处理文本范围：处理到"文档开始"结束
- 当前文档名：示例文档.txt"""
        elif ("抽取实体" in prompt or "entity" in prompt_lower or 
              "从输入文本中抽取所有实体" in prompt or "实体抽取" in prompt):
            return json.dumps([
                {
                    "name": "示例实体1",
                    "content": "这是一个示例实体的描述"
                }
            ], ensure_ascii=False)
        elif ("抽取关系" in prompt or "relation" in prompt_lower or 
              "从输入文本中抽取实体之间的关系" in prompt or "关系抽取" in prompt):
            # 检查实体列表是否为空
            if "已抽取的实体：" in prompt:
                entities_section = prompt.split("已抽取的实体：")[1].split("---")[0].strip()
                # 如果实体部分为空或只有换行符，返回空关系列表
                if not entities_section or entities_section == "\n" or entities_section == "":
                    return json.dumps([], ensure_ascii=False)
            # 如果有实体，返回示例关系（使用与实体抽取一致的实体名称）
            return json.dumps([
                {
                    "entity1_name": "示例实体1",
                    "entity2_name": "示例实体2",
                    "content": "示例实体1与示例实体2之间的关系描述"
                }
            ], ensure_ascii=False)
        elif ("判断.*实体.*匹配" in prompt or "judge.*entity.*match" in prompt_lower or
              "判断新抽取的实体是否与已有实体" in prompt):
            # 模拟实体匹配响应
            return json.dumps({
                "entity_id": "ent_001",
                "need_update": False
            }, ensure_ascii=False)
        elif ("实体后验增强" in prompt or "enhance.*entity.*content" in prompt_lower or
              "对该实体的content进行更细致的补全和挖掘" in prompt or "增强后的完整实体content" in prompt):
            # 模拟实体后验增强响应（JSON格式）
            # 从prompt中提取原始content，然后返回增强后的版本
            if "当前content：" in prompt:
                original_content = prompt.split("当前content：")[1].split("---")[0].strip()
                enhanced_content = f"{original_content}\n\n[增强信息]：基于记忆缓存和当前文本的补充细节和上下文信息。"
            else:
                enhanced_content = "这是一个示例实体的描述\n\n[增强信息]：基于记忆缓存和当前文本的补充细节和上下文信息。"
            # 返回JSON格式
            return json.dumps({"content": enhanced_content}, ensure_ascii=False)
        return "默认响应"
    
    def update_memory_cache(self, current_cache: Optional[MemoryCache], input_text: str,
                           document_name: str = "", text_start_pos: int = 0, 
                           text_end_pos: int = 0, total_text_length: int = 0) -> MemoryCache:
        """
        任务1：更新记忆缓存
        
        Args:
            current_cache: 当前的记忆缓存（如果存在）
            input_text: 当前窗口的输入文本
            document_name: 当前文档名称
            text_start_pos: 当前文本在文档中的起始位置（字符位置）
            text_end_pos: 当前文本在文档中的结束位置（字符位置）
            total_text_length: 文档总长度（字符数）
        
        Returns:
            更新后的新记忆缓存
        """
        system_prompt = """你是一个记忆管理系统。你的任务是维护一个记忆缓存，记录当前处理文档的状态。
记忆缓存应该包含：
1. 当前摘要（详细记录当前窗口的主要内容，包括：当前阅读的具体内容、发生的情况/背景、重要细节等）
2. 自我思考（针对不间断的阅读，思考应该关注哪些内容、预判可能的重点、当前情况下是否有疑虑或需要特别注意的地方等）
3. 系统状态（已处理文本范围、当前文档名等）

重要：
- 当前摘要要详细完善，包括：
  * 当前阅读的具体内容是什么
  * 这些内容是在什么情况下发生的（背景、情境）
  * 有哪些重要的细节、情节、对话、描述等
  * **特别重要**：必须识别并记录文本中的标志性文本锚点，包括但不限于：
    - 章节信息、标题信息、其他结构化标记（如"第一章"、"序言"、"Skills 教程"、"通过第三方云直接安装配置"等单列一行的文本）
    - 这些锚点信息对于定位和分段文本非常重要，必须在摘要中明确记录
- 自我思考部分要体现阅读的主动性和预判能力：
  * 应该关注哪些内容（实体关系、话题发展、关键信息等）
  * 预判可能的重点（后续可能的发展、重要线索、话题知识点走向等）
  * 当前情况下是否有疑虑、困惑或需要特别注意的地方
  * **特别关注**：识别文本中的章节、标题等结构化信息，这些是重要的文本定位锚点
- 系统状态中的"已处理文本范围"应该用自然语言描述当前处理结束的内容，格式：处理到"[结束段落的关键内容或描述]"。例如："处理到'汪淼看到了史强，他倒是一反昨天的粗鲁，向汪淼打招呼'结束"
  * **如果文本中包含章节或标题信息，必须在描述中明确标注**，例如："处理到'第三十六章 标题名称'结束"或"处理到'第一章 开头部分'结束"
- 请用Markdown格式输出，保持简洁但信息完整。"""
        
        # 文本范围信息提示（使用自然语言描述）
        # 注意：这里不再传递字符位置，而是让LLM根据文本内容自然描述
        text_range_hint = "请根据当前输入文本的内容，用自然语言描述当前处理结束的内容，格式：处理到\"[结束段落的关键内容或描述]\"结束。例如：\"处理到'汪淼看到了史强，他倒是一反昨天的粗鲁，向汪淼打招呼'结束\""
        
        if current_cache:
            prompt = f"""当前已有的记忆缓存：

{current_cache.content}

---
新的输入文本：

{input_text}

---

请根据新的输入文本，更新记忆缓存。需要：
1. 保留仍然有用的信息
2. 添加新的信息
3. 删除不再需要的信息
4. 更新当前状态
5. **重要**：当前摘要要详细完善，包括：
   - 当前阅读的具体内容是什么
   - 这些内容是在什么情况下发生的（背景、情境）
   - 有哪些重要的细节、情节、对话、描述等
   - **特别重要**：必须识别并记录文本中的标志性文本锚点，包括但不限于：
     * 章节信息、标题信息、其他结构化标记（如"第一章"、"序言"、"Skills 教程"、"通过第三方云直接安装配置"等单列一行的文本）
     * 这些锚点信息对于定位和分段文本非常重要，必须在摘要中明确记录
6. **重要**：自我思考部分要体现阅读的主动性和预判能力：
   - 应该关注哪些内容（人物关系、情节发展、关键信息等）
   - 预判可能的重点（后续可能的发展、重要线索等）
   - 当前情况下是否有疑虑、困惑或需要特别注意的地方
   - **特别关注**：识别文本中的章节、标题等结构化信息，这些是重要的文本定位锚点
7. **重要**：系统状态中的"已处理文本范围"应该用自然语言描述当前处理结束的内容：{text_range_hint}
   - **如果文本中包含章节或标题信息，必须在描述中明确标注**，例如："处理到'第三十六章 标题名称'结束"

输出更新后的完整记忆缓存内容（Markdown格式）："""
        else:
            prompt = f"""这是第一个记忆缓存。输入文本：

{input_text}

---

**重要**：
1. 当前摘要要详细完善，包括：
   - 当前阅读的具体内容是什么
   - 这些内容是在什么情况下发生的（背景、情境）
   - 有哪些重要的细节、情节、对话、描述等
   - **特别重要**：必须识别并记录文本中的标志性文本锚点，包括但不限于：
     * 章节信息、标题信息、其他结构化标记（如"第一章"、"序言"、"Skills 教程"、"通过第三方云直接安装配置"等单列一行的文本）
     * 这些锚点信息对于定位和分段文本非常重要，必须在摘要中明确记录
2. 自我思考部分要体现阅读的主动性和预判能力：
   - 应该关注哪些内容（人物关系、情节发展、关键信息等）
   - 预判可能的重点（后续可能的发展、重要线索等）
   - 当前情况下是否有疑虑、困惑或需要特别注意的地方
   - **特别关注**：识别文本中的章节、标题等结构化信息，这些是重要的文本定位锚点
3. 系统状态中的"已处理文本范围"应该用自然语言描述当前处理结束的内容：{text_range_hint}
   - **如果文本中包含章节或标题信息，必须在描述中明确标注**，例如："处理到'第三十六章 标题名称'结束"

请创建初始的记忆缓存内容（Markdown格式）："""
        
        new_content = self._call_llm(prompt, system_prompt)
        
        # 清理 markdown 代码块标识符
        new_content = self._clean_markdown_code_blocks(new_content)
        
        # 创建新的MemoryCache
        new_cache_id = f"cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 只保存文档名，不包含路径
        doc_name_only = document_name.split('/')[-1] if document_name else ""
        
        return MemoryCache(
            id=new_cache_id,
            content=new_content,
            physical_time=datetime.now(),
            doc_name=doc_name_only,
            activity_type="文档处理"
        )
    
    def extract_entities(self, memory_cache: MemoryCache, input_text: str,
                         max_iterations: int = 3,
                         enable_iterative: bool = True,
                         verbose: bool = False) -> List[Dict[str, str]]:
        """
        任务2：抽取实体（支持迭代抽取以提高完整性）
        
        Args:
            memory_cache: 当前的记忆缓存
            input_text: 当前窗口的输入文本
            max_iterations: 最大迭代次数（默认3次）
            enable_iterative: 是否启用迭代抽取（默认True）
            verbose: 是否输出详细信息
        
        Returns:
            抽取的实体列表，每个实体包含 name 和 content
        """
        # 如果禁用迭代或文本很短，使用一次性抽取
        if not enable_iterative or len(input_text) < 500:
            return self._extract_entities_single_pass(memory_cache, input_text, None, verbose)
        
        # 迭代抽取策略
        all_entities = []
        # 记录已发现的实体（基于实体名称去重）
        seen_entity_names = set()
        
        if verbose:
            print(f"      开始迭代抽取实体（最大迭代次数: {max_iterations}）")
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"      第 {iteration + 1}/{max_iterations} 轮迭代...")
            
            # 第一轮迭代：一次性抽取所有实体
            if iteration == 0:
                if verbose:
                    print(f"        调用LLM进行初始实体抽取...")
                new_entities = self._extract_entities_single_pass(
                    memory_cache, input_text, None, verbose
                )
                if verbose:
                    print(f"        初始抽取完成，获得 {len(new_entities)} 个实体")
            else:
                # 后续迭代：使用已抽取的实体来查漏补缺
                if verbose:
                    print(f"        使用已抽取的实体进行查漏补缺（已抽取 {len(all_entities)} 个实体）...")
                new_entities = self._extract_entities_single_pass(
                    memory_cache, input_text, all_entities, verbose
                )
                if verbose:
                    print(f"        查漏补缺完成，获得 {len(new_entities)} 个新实体")
            
            # 更新已发现的实体（按实体名称去重）
            new_entities_count = 0
            for entity in new_entities:
                entity_name = entity['name'].strip()
                if entity_name and entity_name not in seen_entity_names:
                    seen_entity_names.add(entity_name)
                    all_entities.append(entity)
                    new_entities_count += 1
            
            if verbose:
                print(f"        第 {iteration + 1} 轮迭代完成，新增 {new_entities_count} 个实体，累计 {len(all_entities)} 个实体")
            
            # 如果本次迭代没有抽取到新实体，停止迭代
            if new_entities_count == 0:
                if verbose:
                    print(f"        未发现新实体，停止迭代")
                break
        
        if verbose:
            print(f"      实体抽取完成，共获得 {len(all_entities)} 个实体")
        
        return all_entities
    
    def _extract_entities_single_pass(self, memory_cache: MemoryCache, input_text: str,
                                     existing_entities: Optional[List[Dict[str, str]]] = None,
                                     verbose: bool = False) -> List[Dict[str, str]]:
        """
        单次实体抽取
        
        Args:
            memory_cache: 记忆缓存
            input_text: 输入文本
            existing_entities: 已抽取的实体列表（用于查漏补缺，可选）
            verbose: 是否输出详细信息
        """
        system_prompt = """你是一个概念实体抽取系统。你的任务是从文本中抽取"概念实体"。

**概念实体的定义：**
概念实体不同于传统的命名实体（人名、地名、组织名），它是一个更广泛的概念，包括：
- **名字**：人物姓名、角色名称、代号等
- **事件**：发生的事件、活动、行动等（如"三体入侵"、"会议"、"战斗"、"见面"等）
- **时间**：时间点、时间段、时间概念等（如"2024年"、"古代"、"未来"等）
- **人物**：人物角色、身份、职业等（如"科学家"、"军官"、"学生"等）
- **地点**：地理位置、场所、空间概念等（如"北京"、"作战中心"、"太空"等）
- **抽象概念**：思想、理论、方法、原则等（如"黑暗森林法则"、"量子理论"、"战略"等）
- **特殊词汇**：专业术语、技术名词、特定概念等（如"智子"、"面壁计划"、"ETO"等）
- **标志性文本锚点**：章节、标题等结构化标记（见下方详细说明）

**概念实体的特点：**
- 可以是一个词，也可以是一个短语
- 可以用括号来丰富定义、状况（如"张三（老师）"、"三体入侵（第一次）"、"北京（城市）"等）
- 应该是有意义的、能够独立理解的概念单元
- 不要抽取指代词（如"他"、"它"、"这个"等）

每个概念实体需要包含：
- name: 实体名称（可以是词或短语，可以用括号丰富定义）
- content: 实体的整体概要性总结，重点描述该实体本身的属性、介绍、特点

**⚠️ 特别重要：标志性文本锚点必须被抽取为实体 ⚠️**
标志性文本锚点是文本中的结构化标记，用于定位和分段文本，**必须被识别并抽取为实体**。这些锚点包括但不限于：
- 章节信息（如"第一章"、"序言"、"Chapter 1"、"Part I"等）
- 标题信息（如"第一部分"、"第二篇"、"Section 2"、"Skills 教程"等）
- 其他结构化标记（如"第一幕"、"第二场"、"通过第三方云直接安装配置"等单列一行的文本）
- 任何单独成行、用于组织文本结构的标记性文本

**标志性文本锚点的识别特征：**
1. 通常单独成行或位于段落开头
2. 具有明显的结构化特征（如数字编号、层级标记等）
3. 用于划分文本的不同部分或章节
4. 在文本中起到定位和导航的作用

**标志性文本锚点实体的要求：**
- name: 使用完整的章节/标题名称，例如："第三十六章"、"第一章 标题名称"、"Skills 教程"等
- content: 必须包含以下信息：
  1. 该章节/标题的完整名称
  2. 在文本中的位置信息（如"位于文本开头"、"位于文本中间部分"等）
  3. 该章节/标题下的主要内容概述
  4. 该章节/标题在整个文档结构中的作用和意义

name命名要求：
1. 使用文本中出现的原始名称，可以是词或短语
2. **重要：使用括号丰富定义和状况**
   - 如果文本中存在同名但实际是不同实体的情况，必须在name中用括号标注区分
     * 例如：同一文本中出现两个"张三"，一个是老师，一个是学生，应分别命名为"张三（老师）"和"张三（学生）"
     * 例如：同一文本中"北京"既指城市又指公司名，应分别命名为"北京（城市）"和"北京（公司）"
   - 可以用括号补充实体的状态、属性、上下文等信息
     * 例如："三体入侵（第一次）"、"会议（紧急）"、"张三（已故）"、"北京（2024年）"等
     * 例如：事件可以标注"事件（发生时间）"、"事件（地点）"等
     * 例如：抽象概念可以标注"理论（提出者）"、"方法（应用场景）"等
   - 对于章节/标题类实体，应使用完整的章节/标题名称作为name，例如："第三十六章"、"第一章 标题名称"等
3. 括号标注应简洁明了，能够清晰区分不同实体或补充重要信息
4. **关键约束**：name必须是content中描述的那个实体的名称，不能是content中提到的其他实体的名称
   - 例如：如果content描述的是"一名军官，跟随汪淼进入作战中心"，那么name应该是这个军官的名字（如果文本中有提到），而不是"汪淼"
   - 例如：如果content描述的是"一名学生，向老师张三请教问题"，那么name应该是这个学生的名字，而不是"张三"
   - 如果content中描述的实体在文本中没有明确的名字，可以使用描述性的名称（如"军官"、"学生"等），但必须确保name与content描述的实体一致

content要求：
1. 聚焦于实体本身的描述，包括实体的类型、属性、特征、特点等
2. 形成一个整体的概要性总结，全面概括该实体的核心信息
3. 重点抽取和识别描述该实体的具体属性、介绍、特点
4. 避免包含实体与其他实体的关系或交互信息，专注于实体自身的描述
5. **对于章节/标题类实体**：content应包含该章节/标题的完整信息、在文本中的位置、该章节/标题下的主要内容概述等

name和content一致性要求：
1. **必须确保**：name和content描述的是同一个实体
2. **禁止**：name是content中提到的其他实体的名称
3. 在生成实体时，先确定content描述的是哪个实体，然后使用该实体的名称作为name

重要：只输出JSON数组，不要包含任何其他文字。每个元素是一个对象，包含name和content字段。
使用英文冒号和逗号，不要使用中文标点符号。"""
        
        # 如果有已抽取的实体，在提示词中说明（用于查漏补缺）
        existing_entities_str = ""
        if existing_entities:
            # 如果实体包含content，也显示content（限制长度）
            existing_entities_list = []
            for e in existing_entities:
                entity_name = e.get('name', '').strip()
                entity_content = e.get('content', '').strip()
                if entity_content:
                    # 限制content长度
                    content_snippet = entity_content[:self.content_snippet_length] + ('...' if len(entity_content) > self.content_snippet_length else '')
                    existing_entities_list.append(f"- {entity_name}：{content_snippet}")
                else:
                    existing_entities_list.append(f"- {entity_name}")
            existing_entities_str = f"""

---

已抽取的实体（不要重复生成）：

{chr(10).join(existing_entities_list)}

注意：请检查是否还有其他实体，不要重复生成已抽取的实体。"""
        
        prompt = f"""当前记忆缓存：

{memory_cache.content}

---

输入文本：

{input_text}{existing_entities_str}

---

请从输入文本中抽取所有概念实体。请严格遵循系统提示中的规则，特别注意：
1. **概念实体包括**：名字、事件、时间、人物、地点、抽象概念、特殊词汇、标志性文本锚点等
2. **必须优先识别并抽取标志性文本锚点**（章节、标题等结构化标记）
3. 如果已有已抽取的实体，不要重复生成，只找出其他实体
4. 尽可能全面地找出所有概念实体，包括词、短语、可以用括号丰富定义的概念等，确保没有遗漏

输出JSON格式的实体列表："""
        
        if verbose:
            if existing_entities:
                print(f"          正在调用LLM查漏补缺（已抽取 {len(existing_entities)} 个实体）...")
            else:
                print(f"          正在调用LLM进行实体抽取...")
        
        response = self._call_llm(prompt, system_prompt)
        
        if verbose:
            print(f"          LLM调用完成，正在解析结果...")
        
        # 解析JSON响应
        try:
            # 尝试提取JSON部分（如果响应包含其他文本）
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            # 清理JSON字符串，修复常见错误
            response = self._clean_json_string(response)
            
            entities = json.loads(response)
            if not isinstance(entities, list):
                entities = [entities]
            
            # 验证并清理实体数据，移除ID字段（如果存在）
            cleaned_entities = []
            for entity in entities:
                if isinstance(entity, dict) and 'name' in entity and 'content' in entity:
                    # 只保留name和content字段，移除其他字段（如entity_id）
                    cleaned_entity = {
                        'name': str(entity['name']).strip(),
                        'content': str(entity['content']).strip()
                    }
                    cleaned_entities.append(cleaned_entity)
            
            if verbose:
                print(f"          解析完成，获得 {len(cleaned_entities)} 个实体")
            
            return cleaned_entities
        except json.JSONDecodeError as e:
            print(f"解析实体JSON失败: {e}")
            print(f"响应内容: {response[:500]}...")  # 只显示前500个字符
            # 尝试修复常见的JSON错误
            try:
                # 尝试修复中文标点符号
                fixed_response = self._fix_json_errors(response)
                entities = json.loads(fixed_response)
                if not isinstance(entities, list):
                    entities = [entities]
                cleaned_entities = []
                for entity in entities:
                    if isinstance(entity, dict) and 'name' in entity and 'content' in entity:
                        cleaned_entity = {
                            'name': str(entity['name']).strip(),
                            'content': str(entity['content']).strip()
                        }
                        cleaned_entities.append(cleaned_entity)
                print(f"已修复JSON错误，成功解析 {len(cleaned_entities)} 个实体")
                return cleaned_entities
            except:
                return []
    
    def extract_entities_by_names(self, memory_cache: MemoryCache, input_text: str,
                                  entity_names: List[str],
                                  verbose: bool = False) -> List[Dict[str, str]]:
        """
        抽取指定名称的实体
        
        Args:
            memory_cache: 记忆缓存
            input_text: 输入文本
            entity_names: 要抽取的实体名称列表
            verbose: 是否输出详细信息
        
        Returns:
            抽取的实体列表，每个实体包含 name 和 content
        """
        if not entity_names:
            return []
        
        system_prompt = """你是一个实体抽取系统。你的任务是从文本中抽取指定的实体。
实体应该是具体的、有意义的对象，不要抽取指代词（如"他"、"它"、"这个"等）。
每个实体需要包含：
- name: 实体名称（必须与指定的名称完全一致）
- content: 实体的整体概要性总结，重点描述该实体本身的属性、介绍、特点

content要求：
1. 聚焦于实体本身的描述，包括实体的类型、属性、特征、特点等
2. 形成一个整体的概要性总结，全面概括该实体的核心信息
3. 重点抽取和识别描述该实体的具体属性、介绍、特点
4. 避免包含实体与其他实体的关系或交互信息，专注于实体自身的描述

重要：只输出JSON数组，不要包含任何其他文字。每个元素是一个对象，包含name和content字段。
使用英文冒号和逗号，不要使用中文标点符号。"""
        
        entity_names_str = "\n".join([f"- {name}" for name in entity_names])
        
        prompt = f"""当前记忆缓存：

{memory_cache.content}

---

输入文本：

{input_text}

---

需要抽取的实体名称列表（必须从输入文本中找到这些实体并抽取）：

{entity_names_str}

---

请从输入文本中抽取上述指定的实体。注意：
1. 只抽取列表中指定的实体，不要抽取其他实体
2. 实体名称必须与列表中的名称完全一致
3. 为每个实体写详细的content描述，重点抽取和识别描述该实体的具体属性、介绍、特点
4. content应该是对实体本身的整体概要性总结，包括实体的类型、属性、特征、特点等
5. 专注于实体自身的描述，避免包含实体与其他实体的关系或交互信息
6. 如果某个实体在文本中找不到，可以跳过（不包含在结果中）

输出JSON格式的实体列表："""
        
        if verbose:
            print(f"          正在调用LLM抽取指定实体（共 {len(entity_names)} 个）...")
        
        response = self._call_llm(prompt, system_prompt)
        
        if verbose:
            print(f"          LLM调用完成，正在解析结果...")
        
        # 解析JSON响应（复用_extract_entities_single_pass的逻辑）
        try:
            # 尝试提取JSON部分（如果响应包含其他文本）
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            # 清理JSON字符串，修复常见错误
            response = self._clean_json_string(response)
            
            entities = json.loads(response)
            if not isinstance(entities, list):
                entities = [entities]
            
            cleaned_entities = []
            for entity in entities:
                if isinstance(entity, dict) and 'name' in entity and 'content' in entity:
                    cleaned_entity = {
                        'name': str(entity['name']).strip(),
                        'content': str(entity['content']).strip()
                    }
                    # 只保留在指定名称列表中的实体
                    if cleaned_entity['name'] in entity_names:
                        cleaned_entities.append(cleaned_entity)
            
            if verbose:
                print(f"          解析完成，获得 {len(cleaned_entities)} 个实体")
            
            return cleaned_entities
        except json.JSONDecodeError as e:
            if verbose:
                print(f"          解析实体JSON失败: {e}")
            # 尝试修复常见的JSON错误
            try:
                fixed_response = self._fix_json_errors(response)
                entities = json.loads(fixed_response)
                if not isinstance(entities, list):
                    entities = [entities]
                cleaned_entities = []
                for entity in entities:
                    if isinstance(entity, dict) and 'name' in entity and 'content' in entity:
                        cleaned_entity = {
                            'name': str(entity['name']).strip(),
                            'content': str(entity['content']).strip()
                        }
                        if cleaned_entity['name'] in entity_names:
                            cleaned_entities.append(cleaned_entity)
                if verbose:
                    print(f"          已修复JSON错误，成功解析 {len(cleaned_entities)} 个实体")
                return cleaned_entities
            except:
                return []
    
    def extract_relations(self, memory_cache: MemoryCache, input_text: str, 
                         entities: Union[List[Dict[str, str]], List[Entity]], 
                         max_iterations: int = 3,
                         absolute_max_iterations: int = 50,
                         enable_iterative: bool = True,
                         verbose: bool = False) -> List[Dict[str, str]]:
        """
        任务3：抽取概念关系边（支持迭代抽取以提高完整性）
        
        Args:
            memory_cache: 当前的记忆缓存
            input_text: 当前窗口的输入文本
            entities: 实体列表，可以是Dict（name, content）或Entity对象（包含entity_id）
            max_iterations: 最大迭代次数（默认3次），超过后会继续运行直到所有实体都有关系边
            absolute_max_iterations: 绝对最大迭代次数（默认50次），超过后强制停止，防止无限循环
            enable_iterative: 是否启用迭代抽取（默认True）
            verbose: 是否输出详细日志
        
        Returns:
            抽取的关系列表，每个关系包含 entity1_name, entity2_name, content
        """
        if not entities:
            return []
        
        # 统一转换为实体信息字典格式
        entity_info_list = []
        for entity in entities:
            if isinstance(entity, Entity):
                entity_info = {
                    'name': entity.name,
                    'content': entity.content,
                    'entity_id': entity.entity_id
                }
            else:
                entity_info = {
                    'name': entity['name'],
                    'content': entity['content'],
                    'entity_id': None
                }
            entity_info_list.append(entity_info)
        
        # 如果实体数量很少（<=3个）或禁用迭代，使用一次性抽取
        if not enable_iterative or len(entity_info_list) <= 3:
            if verbose:
                print(f"      使用单次抽取模式（实体数量: {len(entity_info_list)}）")
            return self._extract_relations_single_pass(
                memory_cache, input_text, entity_info_list, 
                None, None, verbose=verbose
            )
        
        # 迭代抽取策略
        all_relations = []
        # 记录已发现的关系（基于实体对+关系内容的组合）
        seen_relations = set()  # 元素是 (entity1_name, entity2_name, content_hash)
        # 记录当前文本中已抽取关系的实体对（用于后续迭代查漏补缺）
        # key: (entity1_name, entity2_name), value: List[关系内容]
        pairs_with_relations = {}
        
        if verbose:
            print(f"      开始迭代抽取（实体数量: {len(entity_info_list)}，只基于当前文本）")
        
        iteration = 0
        while True:
            iteration += 1
            
            # 检查是否超过绝对最大迭代次数（强制停止，防止无限循环）
            if iteration > absolute_max_iterations:
                if verbose:
                    print(f"      ⚠️ 已达到绝对最大迭代次数 {absolute_max_iterations}，强制停止迭代")
                    # 计算未覆盖的实体
                    all_names = {e['name'].strip() for e in entity_info_list if e.get('name', '').strip()}
                    covered_names = set()
                    for rel in all_relations:
                        e1 = rel.get('entity1_name', '').strip()
                        e2 = rel.get('entity2_name', '').strip()
                        if e1:
                            covered_names.add(e1)
                        if e2:
                            covered_names.add(e2)
                    uncovered_names = all_names - covered_names
                    if uncovered_names:
                        print(f"        注意：仍有 {len(uncovered_names)} 个实体未覆盖: {'| '.join(list(uncovered_names)[:5])}{'...' if len(uncovered_names) > 5 else ''}")
                break
            
            # 判断是否超过最大迭代次数
            exceeded_max_iterations = iteration > max_iterations
            
            if exceeded_max_iterations:
                if verbose:
                    print(f"      第 {iteration} 轮迭代（已超过设定的最大迭代次数 {max_iterations}，继续运行以确保所有实体都有关系边，绝对上限 {absolute_max_iterations}）...")
            else:
                if verbose:
                    print(f"      第 {iteration}/{max_iterations} 轮迭代...")
            
            # 在每一轮迭代前，先对已收集的关系进行去重（除了第一轮）
            if iteration > 1:
                # 对all_relations进行去重，重新构建seen_relations和pairs_with_relations
                if verbose:
                    print(f"        迭代前去重：当前有 {len(all_relations)} 个关系...")
                
                # 重新构建seen_relations和pairs_with_relations（基于去重后的关系）
                seen_relations.clear()
                pairs_with_relations.clear()
                deduplicated_all_relations = []
                
                for rel in all_relations:
                    entity1_name = rel.get('entity1_name', '').strip()
                    entity2_name = rel.get('entity2_name', '').strip()
                    content = rel.get('content', '').strip()
                    
                    if not entity1_name or not entity2_name or not content:
                        continue
                    
                    # 标准化实体对（按字母顺序排序，使关系无向化）
                    normalized_pair = self._normalize_entity_pair(entity1_name, entity2_name)
                    normalized_entity1 = normalized_pair[0]
                    normalized_entity2 = normalized_pair[1]
                    
                    content_hash = hash(content.strip().lower())
                    # 使用标准化后的实体对作为key，使(A,B)和(B,A)被视为同一个关系
                    relation_key = (normalized_entity1, normalized_entity2, content_hash)
                    
                    # 如果这个关系还没有见过，添加到去重后的列表
                    if relation_key not in seen_relations:
                        seen_relations.add(relation_key)
                        pair = (normalized_entity1, normalized_entity2)  # 使用标准化后的实体对
                        if pair not in pairs_with_relations:
                            pairs_with_relations[pair] = []
                        pairs_with_relations[pair].append(content)
                        # 确保关系使用标准化后的实体对
                        rel_copy = rel.copy()
                        rel_copy['entity1_name'] = normalized_entity1
                        rel_copy['entity2_name'] = normalized_entity2
                        deduplicated_all_relations.append(rel_copy)
                
                # 更新all_relations为去重后的版本
                removed_count = len(all_relations) - len(deduplicated_all_relations)
                all_relations = deduplicated_all_relations
                
                if verbose and removed_count > 0:
                    print(f"        去重完成，去除了 {removed_count} 个重复关系，剩余 {len(all_relations)} 个关系")
                elif verbose:
                    print(f"        去重完成，当前有 {len(all_relations)} 个关系（无重复）")
            
            # 计算未覆盖的实体（还没有出现在任何关系中的实体）
            covered_entity_names = set()
            for rel in all_relations:
                entity1_name = rel.get('entity1_name', '').strip()
                entity2_name = rel.get('entity2_name', '').strip()
                if entity1_name:
                    covered_entity_names.add(entity1_name)
                if entity2_name:
                    covered_entity_names.add(entity2_name)
            
            all_entity_names = {e['name'].strip() for e in entity_info_list if e.get('name', '').strip()}
            uncovered_entity_names = list(all_entity_names - covered_entity_names)
            
            if verbose and uncovered_entity_names:
                print(f"        未覆盖的实体（需要优先补全）: {len(uncovered_entity_names)} 个 - {', '.join(uncovered_entity_names[:5])}{'...' if len(uncovered_entity_names) > 5 else ''}")
            elif verbose:
                print(f"        所有实体都已覆盖，可以进行更多元、更细节的关系抽取")
            
            # 检查是否所有实体都有关系边
            all_entities_covered = len(uncovered_entity_names) == 0
            
            # 第一次迭代：一次性抽取所有关系
            if iteration == 1:
                if verbose:
                    print(f"        调用LLM进行初始关系抽取（只基于当前文本）...")
                new_relations = self._extract_relations_single_pass(
                    memory_cache, input_text, entity_info_list,
                    None, uncovered_entity_names, verbose=verbose
                )
                if verbose:
                    print(f"        初始抽取完成，获得 {len(new_relations)} 个关系（LLM生成）")
                
                # 对第一轮的关系也进行去重（虽然seen_relations是空的，但可以去除重复的关系）
                deduplicated_relations = self._deduplicate_relations(new_relations, seen_relations)
                
                if verbose and len(new_relations) != len(deduplicated_relations):
                    print(f"        去重后剩余 {len(deduplicated_relations)} 个关系（去除了 {len(new_relations) - len(deduplicated_relations)} 个重复关系）")
                
                new_relations = deduplicated_relations
            else:
                # 后续迭代：如果有未覆盖的实体，不传入已生成的关系列表；只有全部覆盖了，才传入已生成的关系列表
                if all_entities_covered:
                    # 所有实体都已覆盖，传入已生成的关系列表进行查漏补缺
                    if verbose:
                        print(f"        使用已抽取的关系进行查漏补缺（已抽取 {len(all_relations)} 个关系）...")
                    new_relations = self._extract_relations_single_pass(
                        memory_cache, input_text, entity_info_list,
                        pairs_with_relations, uncovered_entity_names, verbose=verbose
                    )
                else:
                    # 还有未覆盖的实体，不传入已生成的关系列表，优先补全未覆盖的实体
                    if verbose:
                        print(f"        优先补全未覆盖的实体（已抽取 {len(all_relations)} 个关系，但还有 {len(uncovered_entity_names)} 个实体未覆盖）...")
                    new_relations = self._extract_relations_single_pass(
                        memory_cache, input_text, entity_info_list,
                        None, uncovered_entity_names, verbose=verbose
                    )
                
                if verbose:
                    print(f"        查漏补缺完成，获得 {len(new_relations)} 个关系（LLM生成）")
            
            # 对LLM生成的关系进行代码层面的去重（避免重复的关系传入下一轮）
            deduplicated_relations = self._deduplicate_relations(new_relations, seen_relations)
            
            if verbose and len(new_relations) != len(deduplicated_relations):
                print(f"        去重后剩余 {len(deduplicated_relations)} 个新关系（去除了 {len(new_relations) - len(deduplicated_relations)} 个重复关系）")
            
            # 更新已发现的关系和实体对关系映射
            new_relations_count = 0
            for rel in deduplicated_relations:
                entity1_name = rel['entity1_name']
                entity2_name = rel['entity2_name']
                content = rel.get('content', '')
                
                # 标准化实体对（按字母顺序排序，使关系无向化）
                normalized_pair = self._normalize_entity_pair(entity1_name, entity2_name)
                normalized_entity1 = normalized_pair[0]
                normalized_entity2 = normalized_pair[1]
                
                # 使用关系内容的哈希值来去重（允许同一实体对存在多个不同关系）
                content_hash = hash(content.strip().lower())
                # 使用标准化后的实体对作为key，使(A,B)和(B,A)被视为同一个关系
                relation_key = (normalized_entity1, normalized_entity2, content_hash)
                
                if relation_key not in seen_relations:
                    seen_relations.add(relation_key)
                    pair = (normalized_entity1, normalized_entity2)  # 使用标准化后的实体对
                    if pair not in pairs_with_relations:
                        pairs_with_relations[pair] = []
                    pairs_with_relations[pair].append(content)
                    # 确保关系使用标准化后的实体对
                    rel_copy = rel.copy()
                    rel_copy['entity1_name'] = normalized_entity1
                    rel_copy['entity2_name'] = normalized_entity2
                    all_relations.append(rel_copy)
                    new_relations_count += 1
            
            if verbose:
                print(f"        第 {iteration} 轮迭代完成，新增 {new_relations_count} 个关系，累计 {len(all_relations)} 个关系")
            
            # 重新计算未覆盖的实体（在添加新关系后）
            covered_entity_names_after = set()
            for rel in all_relations:
                entity1_name = rel.get('entity1_name', '').strip()
                entity2_name = rel.get('entity2_name', '').strip()
                if entity1_name:
                    covered_entity_names_after.add(entity1_name)
                if entity2_name:
                    covered_entity_names_after.add(entity2_name)
            
            uncovered_entity_names_after = list(all_entity_names - covered_entity_names_after)
            all_entities_covered_after = len(uncovered_entity_names_after) == 0
            
            # 停止条件：
            # 1. 如果所有实体都有关系边，且已达到或超过最大轮次，停止迭代
            # 2. 如果所有实体都有关系边，但还没达到最大轮次，继续迭代直到达到最大轮次
            # 3. 如果还有实体没有关系边，继续迭代（即使超过最大迭代次数）
            if all_entities_covered_after:
                # 所有实体都已覆盖
                if iteration >= max_iterations:
                    # 已达到或超过最大轮次，停止迭代
                    if exceeded_max_iterations:
                        if verbose:
                            print(f"        所有实体都已覆盖，停止迭代（共运行了 {iteration} 轮，超过设定的 {max_iterations} 轮）")
                    else:
                        if verbose:
                            print(f"        所有实体都已覆盖，且已达到最大轮次 {max_iterations}，停止迭代")
                    break
                else:
                    # 所有实体都已覆盖，但还没达到最大轮次，继续迭代
                    if verbose:
                        print(f"        所有实体都已覆盖，但未达到最大轮次 {max_iterations}，继续迭代以抽取更多关系...")
                    # 继续循环，直到达到最大轮次
            else:
                # 如果还有未覆盖的实体，继续迭代（即使超过最大迭代次数）
                if exceeded_max_iterations:
                    if verbose:
                        print(f"        仍有 {len(uncovered_entity_names_after)} 个实体未覆盖，继续迭代（已超过最大迭代次数 {max_iterations}，目标：确保所有实体都有关系边）...")
                elif new_relations_count == 0:
                    if verbose:
                        print(f"        仍有 {len(uncovered_entity_names_after)} 个实体未覆盖，但本次迭代未发现新关系，继续尝试...")
                # 继续循环
        
        if verbose:
            print(f"      关系抽取完成，共获得 {len(all_relations)} 个关系")
        
        return all_relations
    
    def _extract_relations_single_pass(self, memory_cache: MemoryCache, 
                                      input_text: str, 
                                      entities: List[Dict[str, Any]],
                                      existing_relations: Optional[Dict[tuple, List[str]]] = None,
                                      uncovered_entities: Optional[List[str]] = None,
                                      verbose: bool = False) -> List[Dict[str, str]]:
        """
        单次关系抽取
        
        Args:
            memory_cache: 记忆缓存
            input_text: 输入文本
            entities: 实体信息列表（包含name, content, entity_id）
            existing_relations: 已抽取的关系字典，key是(entity1, entity2)，value是关系content列表
            uncovered_entities: 未覆盖的实体名称列表（还没有任何关系，需要优先补全）
            verbose: 是否输出详细日志
        """
        system_prompt = """你是一个关系抽取系统。从文本中抽取概念实体之间的关系。

输出JSON数组，每个关系包含：
- entity1_name: 概念实体名称
- entity2_name: 概念实体名称  
- content: 关系描述（简洁明了，包含关系类型、性质、发生条件等）

⚠️ 极其重要 - 实体名称必须完全一致：
- entity1_name和entity2_name必须与提供的概念实体列表中的名称【完全一致】
- 包括括号及其中的说明文字，如"丁仪（物理学家）"不能简化为"丁仪"
- 不能省略、简化、修改或重新措辞任何实体名称
- 复制粘贴实体列表中的名称，不要自行创造

注意：关系是无向的，只输出JSON数组，使用英文标点。"""
        
        # 构建实体名称集合，用于区分已覆盖和未覆盖的实体
        uncovered_set = set(uncovered_entities) if uncovered_entities else set()
        
        # 辅助函数：格式化实体信息
        def format_entity(e: Dict[str, Any]) -> str:
            name = e.get('name', '').strip()
            content = e.get('content', '').strip()
            if content:
                snippet = content[:self.content_snippet_length] + ('...' if len(content) > self.content_snippet_length else '')
                return f"- {name}：{snippet}"
            return f"- {name}"
        
        # 分离未覆盖实体和已覆盖实体
        uncovered_entities_list = []
        covered_entities_list = []
        for e in entities:
            name = e.get('name', '').strip()
            if not name:
                continue
            if name in uncovered_set:
                uncovered_entities_list.append(format_entity(e))
            else:
                covered_entities_list.append(format_entity(e))
        
        # 构建prompt各部分
        prompt_parts = [f"记忆缓存：\n{memory_cache.content}"]
        
        # 已抽取的关系（如果有，且所有实体都已覆盖）
        if existing_relations:
            relations_info = []
            for (e1, e2), contents in existing_relations.items():
                if contents:
                    relations_info.append(f"- {e1} ↔ {e2}：{' / '.join(contents)}")
            if relations_info:
                prompt_parts.append(f"已抽取的关系（不要重复）：\n" + "\n".join(relations_info))
        
        # 输入文本
        prompt_parts.append(f"输入文本：\n{input_text}")
        
        # 优先实体放在最后（权重更高）
        if uncovered_entities_list:
            uncovered_names = [e.get('name', '').strip() for e in entities if e.get('name', '').strip() in uncovered_set]
            prompt_parts.append("⚠️ 重点：以下概念实体还没有关系，必须优先为它们找到关系：\n\n" + "\n".join(uncovered_names) + "\n\n【注意】输出时entity1_name和entity2_name其中一个必须与上面的概念实体名称【完全一致】，包括括号中的说明文字，不能简化！")
        
        # 简洁的任务说明
        task_instruction = "请抽取概念实体间的关系。"
        if existing_relations:
            task_instruction += "不要重复已有关系。"
        prompt_parts.append(task_instruction)
        
        prompt = "\n\n---\n\n".join(prompt_parts)
        
        if verbose:
            print(f"          正在调用LLM进行关系抽取（实体数量: {len(entities)}）...")
        response = self._call_llm(prompt, system_prompt)
        if verbose:
            print(f"          LLM调用完成，正在解析结果...")
        result = self._parse_relations_response(response)
        
        # 对实体名称进行规范化，将LLM可能简化的名称映射回原始完整名称
        valid_entity_names = {e.get('name', '').strip() for e in entities if e.get('name', '').strip()}
        normalized_count = 0
        for rel in result:
            original_e1 = rel.get('entity1_name', '')
            original_e2 = rel.get('entity2_name', '')
            normalized_e1 = self._normalize_entity_name_to_original(original_e1, valid_entity_names)
            normalized_e2 = self._normalize_entity_name_to_original(original_e2, valid_entity_names)
            if normalized_e1 != original_e1 or normalized_e2 != original_e2:
                rel['entity1_name'] = normalized_e1
                rel['entity2_name'] = normalized_e2
                normalized_count += 1
        
        if verbose:
            if normalized_count > 0:
                print(f"          解析完成，获得 {len(result)} 个关系（规范化了 {normalized_count} 个实体名称）")
            else:
                print(f"          解析完成，获得 {len(result)} 个关系")
        return result
    
    def _extract_relations_for_pairs(self, memory_cache: MemoryCache, 
                                     input_text: str,
                                     entity_pairs: List[tuple]) -> List[Dict[str, str]]:
        """
        针对特定实体对进行关系抽取（补充抽取）
        
        Args:
            memory_cache: 当前的记忆缓存
            input_text: 当前窗口的输入文本
            entity_pairs: 实体对列表，每个元素是 (entity1_name, entity2_name) 的元组
        
        Returns:
            抽取的关系列表
        """
        if not entity_pairs:
            return []
        
        system_prompt = f"""你是一个关系抽取系统。你的任务是从文本中检查特定实体对之间是否存在关系。
{self.RELATION_EXTRACTION_CORE_RULES}
重要：只输出JSON数组，不要包含任何其他文字。
如果某个实体对之间没有关系，不要包含在结果中。
使用英文冒号和逗号，不要使用中文标点符号。"""
        
        # 构建实体对列表字符串
        pairs_str = "\n".join([f"- {pair[0]} 和 {pair[1]}" for pair in entity_pairs])
        
        prompt = f"""当前记忆缓存：

{memory_cache.content}

---

需要检查的实体对：

{pairs_str}

---

输入文本：

{input_text}

---

请仔细检查上述实体对之间是否存在关系。注意：
1. 关系应该是概念性的、有意义的描述
2. content应该专注于描述两个实体之间的具体关系、详细信息以及发生条件
3. 包括关系的类型、性质、特点、内容、特征、表现等详细信息
4. 如果文本中有相关信息，包含关系的发生条件（如时间、地点、情境等）
5. 描述应该简洁明了，不宜过长，重点突出关系的核心信息
6. 只抽取在文本中明确出现或可以推断出的关系
7. 如果某个实体对之间没有关系，不要包含在结果中

输出JSON格式的关系列表（只包含确实存在关系的实体对）："""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_relations_response(response)
    
    def _extract_relations_for_pairs_with_existing(self, memory_cache: MemoryCache, 
                                                   input_text: str,
                                                   entity_pairs: List[tuple],
                                                   existing_relations: Dict[tuple, List[str]],
                                                   verbose: bool = False) -> List[Dict[str, str]]:
        """
        针对特定实体对进行关系抽取，考虑已有关系（用于发现同一实体对的多种关系）
        
        Args:
            memory_cache: 当前的记忆缓存
            input_text: 当前窗口的输入文本
            entity_pairs: 实体对列表，每个元素是 (entity1_name, entity2_name) 的元组
            existing_relations: 已有关系字典，key是实体对，value是已有关系内容列表
        
        Returns:
            抽取的新关系列表（不包括已有关系）
        """
        if not entity_pairs:
            return []
        
        system_prompt = f"""你是一个关系抽取系统。你的任务是从文本中检查特定实体对之间是否存在关系。
{self.RELATION_EXTRACTION_CORE_RULES}
重要：
1. 只输出JSON数组，不要包含任何其他文字
2. 如果某个实体对之间没有关系，不要包含在结果中
3. 如果某个实体对已有关系，只输出与已有关系不同的新关系
4. 两个实体之间可能存在多种不同的关系，请仔细检查
5. 使用英文冒号和逗号，不要使用中文标点符号"""
        
        # 构建实体对列表字符串，包含已有关系信息（包括数据库中已有的和本轮已抽取的）
        pairs_info = []
        for pair in entity_pairs:
            pair_str = f"- {pair[0]} 和 {pair[1]}"
            existing = existing_relations.get(pair, [])
            if existing:
                existing_str = "\n  已抽取的关系（不要重复生成）：" + "\n  ".join([f"  • {rel}" for rel in existing])
                pair_str += existing_str
            pairs_info.append(pair_str)
        
        pairs_str = "\n".join(pairs_info)
        
        prompt = f"""当前记忆缓存：

{memory_cache.content}

---

需要检查的实体对（部分实体对已有已抽取的关系，请找出是否还有其他关系）：

{pairs_str}

---

输入文本：

{input_text}

---

请仔细检查上述实体对之间是否存在关系。注意：
1. 关系应该是概念性的、有意义的描述
2. content应该专注于描述两个实体之间的具体关系、详细信息以及发生条件
3. 包括关系的类型、性质、特点、内容、特征、表现等详细信息
4. 如果文本中有相关信息，包含关系的发生条件（如时间、地点、情境等）
5. 描述应该简洁明了，不宜过长，重点突出关系的核心信息
6. 只抽取在文本中明确出现或可以推断出的关系
7. 如果某个实体对之间没有关系，不要包含在结果中
8. **重要**：对于已有关系的实体对，请检查是否还存在其他不同的关系
9. **重要**：不要重复生成已有关系，只输出与已有关系不同的新关系
10. 两个实体之间可能存在多种不同的关系（例如：既是朋友，又是同事）

输出JSON格式的关系列表（只包含确实存在且与已有关系不同的新关系）："""
        
        if verbose:
            print(f"            正在调用LLM检查 {len(entity_pairs)} 个实体对...")
        response = self._call_llm(prompt, system_prompt)
        if verbose:
            print(f"            LLM调用完成，正在解析结果...")
        result = self._parse_relations_response(response)
        if verbose:
            print(f"            解析完成，获得 {len(result)} 个新关系")
        return result
    
    
    def _get_all_entity_pairs(self, entities: List[Dict[str, str]]) -> List[tuple]:
        """
        获取所有可能的实体对列表（不包括自己和自己）
        
        Args:
            entities: 实体列表（Dict格式，包含name）
        
        Returns:
            所有实体对列表
        """
        entity_names = [e['name'] for e in entities]
        all_pairs = []
        
        # 生成所有可能的实体对（不包括自己和自己）
        for i, entity1 in enumerate(entity_names):
            for j, entity2 in enumerate(entity_names):
                if i != j:  # 不包括自己和自己
                    all_pairs.append((entity1, entity2))
        
        return all_pairs
    
    def _deduplicate_relations(self, relations: List[Dict[str, str]], 
                               seen_relations: set) -> List[Dict[str, str]]:
        """
        对关系进行去重（代码层面的去重规则）
        关系是无向的，将(A,B)和(B,A)视为同一个关系
        
        Args:
            relations: 关系列表
            seen_relations: 已见过的关系集合，元素是 (entity1_name, entity2_name, content_hash)
        
        Returns:
            去重后的关系列表
        """
        deduplicated = []
        for rel in relations:
            entity1_name = rel.get('entity1_name', '').strip()
            entity2_name = rel.get('entity2_name', '').strip()
            content = rel.get('content', '').strip()
            
            if not entity1_name or not entity2_name or not content:
                continue
            
            # 标准化实体对（按字母顺序排序，使关系无向化）
            normalized_pair = self._normalize_entity_pair(entity1_name, entity2_name)
            normalized_entity1 = normalized_pair[0]
            normalized_entity2 = normalized_pair[1]
            
            # 使用关系内容的哈希值来去重（允许同一实体对存在多个不同关系）
            content_hash = hash(content.lower())
            # 使用标准化后的实体对作为key，使(A,B)和(B,A)被视为同一个关系
            relation_key = (normalized_entity1, normalized_entity2, content_hash)
            
            # 如果这个关系还没有见过，添加到结果中
            if relation_key not in seen_relations:
                # 确保输出的关系使用标准化后的实体对
                rel_copy = rel.copy()
                rel_copy['entity1_name'] = normalized_entity1
                rel_copy['entity2_name'] = normalized_entity2
                deduplicated.append(rel_copy)
        
        return deduplicated
    
    def _get_all_entity_pairs_from_info(self, entity_info_list: List[Dict[str, Any]]) -> List[tuple]:
        """
        从实体信息列表获取所有可能的实体对列表（不包括自己和自己）
        
        Args:
            entity_info_list: 实体信息列表（包含name）
        
        Returns:
            所有实体对列表
        """
        entity_names = [e['name'] for e in entity_info_list]
        all_pairs = []
        
        # 生成所有可能的实体对（不包括自己和自己）
        for i, entity1 in enumerate(entity_names):
            for j, entity2 in enumerate(entity_names):
                if i != j:  # 不包括自己和自己
                    all_pairs.append((entity1, entity2))
        
        return all_pairs
    
    def _parse_relations_response(self, response: str) -> List[Dict[str, str]]:
        """
        解析关系抽取的LLM响应
        
        Args:
            response: LLM的响应文本
        
        Returns:
            解析后的关系列表
        """
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            # 清理JSON字符串
            response = self._clean_json_string(response)
            
            relations = json.loads(response)
            # 确保返回的是列表
            if not isinstance(relations, list):
                relations = [relations]
            
            # 验证并清理关系数据，移除ID字段（如果存在）
            # 支持两种格式：entity1_name/entity2_name（无向关系）和entity1_name/entity2_name（向后兼容）
            valid_relations = []
            for rel in relations:
                if not isinstance(rel, dict):
                    print(f"警告：跳过无效的关系格式: {rel}")
                    continue
                
                # 尝试获取实体名称（支持多种格式：英文、中文）
                entity1 = None
                entity2 = None
                
                # 支持英文键名
                if 'entity1_name' in rel and 'entity2_name' in rel:
                    # 新格式：无向关系
                    entity1 = str(rel['entity1_name']).strip()
                    entity2 = str(rel['entity2_name']).strip()
                elif 'entity1_name' in rel and 'entity2_name' in rel:
                    # 旧格式：有向关系（向后兼容）
                    entity1 = str(rel['entity1_name']).strip()
                    entity2 = str(rel['entity2_name']).strip()
                # 支持中文键名
                elif '实体1' in rel and '实体2' in rel:
                    entity1 = str(rel['实体1']).strip()
                    entity2 = str(rel['实体2']).strip()
                elif '实体A' in rel and '实体B' in rel:
                    entity1 = str(rel['实体A']).strip()
                    entity2 = str(rel['实体B']).strip()
                elif 'from' in rel and 'to' in rel:
                    entity1 = str(rel['from']).strip()
                    entity2 = str(rel['to']).strip()
                else:
                    print(f"警告：跳过无效的关系格式（缺少实体名称）: {rel}")
                    continue
                
                if not entity1 or not entity2:
                    print(f"警告：跳过无效的关系格式（实体名称为空）: {rel}")
                    continue
                
                # 标准化实体对（按字母顺序排序，使关系无向化）
                normalized_pair = self._normalize_entity_pair(entity1, entity2)
                
                # 获取content（支持英文和中文键名）
                content = ''
                if 'content' in rel:
                    content = str(rel['content']).strip()
                elif '内容' in rel:
                    content = str(rel['内容']).strip()
                elif '关系内容' in rel:
                    content = str(rel['关系内容']).strip()
                elif '描述' in rel:
                    content = str(rel['描述']).strip()
                
                # 只保留必需的字段，移除其他字段（如relation_id）
                cleaned_relation = {
                    'entity1_name': normalized_pair[0],  # 使用标准化后的顺序
                    'entity2_name': normalized_pair[1],
                    'content': content
                }
                valid_relations.append(cleaned_relation)
            return valid_relations
        except json.JSONDecodeError as e:
            print(f"解析关系JSON失败: {e}")
            print(f"响应内容: {response[:500]}...")  # 只显示前500个字符
            # 尝试修复常见的JSON错误
            try:
                fixed_response = self._fix_json_errors(response)
                relations = json.loads(fixed_response)
                if not isinstance(relations, list):
                    relations = [relations]
                valid_relations = []
                for rel in relations:
                    if not isinstance(rel, dict):
                        continue
                    
                    # 尝试获取实体名称（支持多种格式：英文、中文）
                    entity1 = None
                    entity2 = None
                    
                    # 支持英文键名
                    if 'entity1_name' in rel and 'entity2_name' in rel:
                        entity1 = str(rel['entity1_name']).strip()
                        entity2 = str(rel['entity2_name']).strip()
                    elif 'entity1_name' in rel and 'entity2_name' in rel:
                        entity1 = str(rel['entity1_name']).strip()
                        entity2 = str(rel['entity2_name']).strip()
                    # 支持中文键名
                    elif '实体1' in rel and '实体2' in rel:
                        entity1 = str(rel['实体1']).strip()
                        entity2 = str(rel['实体2']).strip()
                    elif '实体A' in rel and '实体B' in rel:
                        entity1 = str(rel['实体A']).strip()
                        entity2 = str(rel['实体B']).strip()
                    elif 'from' in rel and 'to' in rel:
                        entity1 = str(rel['from']).strip()
                        entity2 = str(rel['to']).strip()
                    else:
                        continue
                    
                    if not entity1 or not entity2:
                        continue
                    
                    # 标准化实体对（按字母顺序排序，使关系无向化）
                    normalized_pair = self._normalize_entity_pair(entity1, entity2)
                    
                    # 获取content（支持英文和中文键名）
                    content = ''
                    if 'content' in rel:
                        content = str(rel['content']).strip()
                    elif '内容' in rel:
                        content = str(rel['内容']).strip()
                    elif '关系内容' in rel:
                        content = str(rel['关系内容']).strip()
                    elif '描述' in rel:
                        content = str(rel['描述']).strip()
                    
                    cleaned_relation = {
                        'entity1_name': normalized_pair[0],
                        'entity2_name': normalized_pair[1],
                        'content': content
                    }
                    valid_relations.append(cleaned_relation)
                print(f"已修复JSON错误，成功解析 {len(valid_relations)} 个关系")
                return valid_relations
            except:
                return []
    
    def judge_entity_match(self, extracted_entity: Dict[str, str], 
                          existing_entities: List[Dict[str, str]],
                          memory_cache: Optional[MemoryCache] = None) -> Optional[Dict[str, Any]]:
        """
        判断抽取的实体是否与已有实体匹配
        
        Args:
            extracted_entity: 抽取的实体（包含name和content）
            existing_entities: 已有实体列表（每个包含entity_id, name, content）
            memory_cache: 当前记忆缓存（可选，用于提供上下文信息）
        
        Returns:
            如果匹配，返回 {"entity_id": "...", "need_update": True/False}
            如果不匹配，返回 None
        """
        if not existing_entities:
            return None
        
        system_prompt = f"""你是一个实体对齐系统。判断新抽取的实体是否与已有实体是**同一个对象**。

{self.ENTITY_ALIGNMENT_CORE_PRINCIPLES}

**输出说明**：
- need_update=true 表示新content有补充信息
- 匹配返回 {{"entity_id": "...", "need_update": true/false}}，不匹配返回 null"""
        
        # 构建记忆缓存信息（如果提供）
        memory_context = ""
        if memory_cache:
            # MemoryCache的content字段包含Markdown格式的完整描述
            # 截取前500字符作为上下文，避免prompt过长
            cache_content = memory_cache.content if memory_cache.content else '无'
            memory_context = f"""
---

当前记忆缓存上下文（用于辅助判断）：
{cache_content}

---
"""
        
        existing_str = "\n\n".join([
            f"- entity_id: {e['entity_id']}, name: {e['name']}, content: {e['content'][:self.content_snippet_length]}{'...' if len(e['content']) > self.content_snippet_length else ''}"
            for e in existing_entities
        ])
        
        extracted_content = extracted_entity['content']
        extracted_content_snippet = extracted_content[:self.content_snippet_length] + ('...' if len(extracted_content) > self.content_snippet_length else '')
        prompt = f"""新抽取的实体：
- name: {extracted_entity['name']}
- content: {extracted_content_snippet}

---

已有实体列表：

{existing_str}

---

判断新实体是否与某个已有实体是**同一个对象**。

⚠️ 三步检查：
1. **名称**：相同、相似、或别名关系？
2. **类型**：都是人物？都是概念？都是作品？（类型不同则不匹配）
3. **主体**：content描述的是同一个具体对象吗？（新信息≠不同对象）

💡 关键区分：
- 同一人的不同事迹 → 匹配（是补充信息）
- 相关但不同类型的对象 → 不匹配（如：人物vs领域、作者vs作品）

匹配返回：{{"entity_id": "...", "need_update": true/false}}
不匹配返回：null"""
        
        response = self._call_llm(prompt, system_prompt)
        
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            result = json.loads(response)
            # 处理不同的返回格式
            if result is None or result == "null":
                return None
            # 如果返回的是列表，取第一个元素（可能是LLM返回了数组格式）
            if isinstance(result, list):
                if len(result) > 0:
                    result = result[0]
                else:
                    return None
            # 确保返回的是字典格式
            if isinstance(result, dict):
                return result
            else:
                return None
        except json.JSONDecodeError:
            return None
    
    def judge_content_need_update(self, old_content: str, new_content: str) -> bool:
        """
        判断内容是否需要更新
        
        比较最新版本的content和当前抽取的content，判断是否需要更新
        
        判断规则：
        1. 如果新内容的所有信息都已经被旧内容包含，返回False（不需要更新）
        2. 如果新内容包含新信息、修正了旧内容、或与旧内容有实质性差异，返回True（需要更新）
        
        Args:
            old_content: 数据库中最新版本的content
            new_content: 当前抽取的content
        
        Returns:
            True表示需要更新，False表示不需要更新
        """
        # 如果内容完全相同，不需要更新
        if old_content.strip() == new_content.strip():
            return False
        
        # 使用LLM判断新内容是否已经被旧内容包含
        system_prompt = """你是一个内容比较系统。你的任务是判断新版本的内容是否已经被旧版本的内容包含，或者是否需要更新。

判断规则：
1. 如果新内容的所有信息都已经被旧内容包含（新内容只是旧内容的子集或重复），返回false（不需要更新）
2. 如果新内容包含新信息、修正了旧内容、或与旧内容有实质性差异，返回true（需要更新）

重要：如果旧内容已经包含了新内容的所有信息，即使表述方式不同，也应该返回false。"""
        
        prompt = f"""数据库中最新版本的内容：

{old_content}

---

当前抽取的内容：

{new_content}

---

请判断当前抽取的内容是否已经被数据库中最新版本的内容包含。

判断标准：
1. **如果新内容的所有信息都已经被旧内容包含**（新内容只是旧内容的子集、重复或换一种说法），返回false（不需要更新）
2. **如果新内容包含新信息**（旧内容中没有的信息），返回true（需要更新）
3. **如果新内容修正了旧内容**（纠正了错误、更新了过时信息），返回true（需要更新）
4. **如果新内容与旧内容有实质性差异**（描述了不同的方面或状态），返回true（需要更新）

只输出true或false，不要其他文字："""
        
        response = self._call_llm(prompt, system_prompt)
        
        # 解析响应
        response = response.strip().lower()
        if response == "true":
            return True
        elif response == "false":
            return False
        else:
            # 如果LLM返回的不是true/false，默认返回True（保守策略，倾向于更新）
            return True
    
    def merge_entity_content(self, old_content: str, new_content: str) -> str:
        """
        合并实体内容（更新时使用）
        
        Args:
            old_content: 旧的内容
            new_content: 新的内容
        
        Returns:
            合并后的内容
        """
        system_prompt = """你是一个内容总结系统。你的任务是基于旧版本和新版本的内容，重新总结生成一个完整、准确的实体描述。

重要：
- **必须只输出JSON格式，包含content字段，不要包含任何其他文字或markdown代码块**"""
        
        prompt = f"""旧版本内容：

{old_content}

---

新版本内容：

{new_content}

---

请基于上述两个版本的内容，重新总结生成一个完整、准确的实体描述。

重要要求：
{self.CONTENT_MERGE_REQUIREMENTS}

**重要：只输出JSON格式，格式如下：**
{{"content": "重新总结后的完整实体描述"}}

不要包含任何其他文字、说明或markdown代码块，只输出纯JSON。"""
        
        response = self._call_llm(prompt, system_prompt)
        
        # 尝试解析JSON响应
        try:
            # 尝试提取JSON部分（如果响应包含其他文本）
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            # 清理JSON字符串，修复常见错误
            response = self._clean_json_string(response)
            
            # 解析JSON
            result = json.loads(response)
            
            # 提取content字段
            if isinstance(result, dict) and 'content' in result:
                merged_content = str(result['content']).strip()
                # 如果content不为空，返回合并后的内容
                if merged_content:
                    return merged_content
            
            # 如果JSON格式不正确或content为空，回退到原始响应
            print(f"警告：实体内容合并返回的JSON格式不正确或content为空，使用原始响应")
            return response.strip()
            
        except json.JSONDecodeError as e:
            # JSON解析失败，尝试修复
            try:
                fixed_response = self._fix_json_errors(response)
                result = json.loads(fixed_response)
                if isinstance(result, dict) and 'content' in result:
                    merged_content = str(result['content']).strip()
                    if merged_content:
                        return merged_content
                print(f"警告：实体内容合并JSON解析失败（已尝试修复），使用原始响应: {e}")
                return response.strip()
            except:
                # 如果修复也失败，返回原始响应（去除可能的markdown代码块）
                print(f"警告：实体内容合并JSON解析失败，使用原始响应: {e}")
                cleaned_response = self._clean_markdown_code_blocks(response)
                return cleaned_response.strip()
    
    def merge_entity_name(self, old_name: str, new_name: str) -> str:
        """
        合并实体名称（更新时使用）
        
        如果新名称与旧名称不同，生成一个包含别称的名称。
        例如：旧名称"科幻世界"，新名称"科幻世界出版机构"，合并后可能是"科幻世界（出版机构）"
        
        Args:
            old_name: 旧的名称
            new_name: 新的名称
        
        Returns:
            合并后的名称
        """
        # 如果名称相同，直接返回
        if old_name == new_name:
            return old_name
        
        # 如果一个名称包含另一个，使用较长的
        if old_name in new_name:
            return new_name
        if new_name in old_name:
            return old_name
        
        system_prompt = """你是一个实体名称合并系统。你的任务是将两个可能指向同一实体的名称合并为一个规范名称。

规则：
1. 选择一个最常用或最规范的名称作为主名称
2. 如果另一个名称是别称、简称或不同角度的描述，用括号附加在主名称后
3. 括号内的内容应该简洁，只保留关键区分信息
4. 如果两个名称差异很大但确实指向同一实体，可以用"又名"或"/"连接

示例：
- "科幻世界" + "科幻世界出版机构" = "科幻世界（出版机构）"
- "刘慈欣" + "大刘" = "刘慈欣（大刘）"
- "北京" + "北京市" = "北京"
- "基石" + "中国科幻基石丛书" = "中国科幻基石丛书（基石）"

重要：只输出JSON格式，包含name字段，不要包含任何其他文字。"""
        
        prompt = f"""旧名称：{old_name}
新名称：{new_name}

请将这两个名称合并为一个规范名称。

**重要：只输出JSON格式，格式如下：**
{{"name": "合并后的规范名称"}}

不要包含任何其他文字、说明或markdown代码块，只输出纯JSON。"""
        
        response = self._call_llm(prompt, system_prompt)
        
        # 尝试解析JSON响应
        try:
            # 尝试提取JSON部分
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            response = self._clean_json_string(response)
            result = json.loads(response)
            
            if isinstance(result, dict) and 'name' in result:
                merged_name = str(result['name']).strip()
                if merged_name:
                    return merged_name
            
            # JSON格式不正确，使用简单合并策略
            return f"{old_name}（{new_name}）"
            
        except json.JSONDecodeError:
            # JSON解析失败，使用简单合并策略
            # 选择较短的作为主名称，较长的作为补充
            if len(old_name) <= len(new_name):
                return f"{old_name}（{new_name}）"
            else:
                return f"{new_name}（{old_name}）"
    
    def judge_relation_match(self, extracted_relation: Dict[str, str],
                            existing_relations: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """
        判断抽取的关系是否与已有关系匹配
        
        Args:
            extracted_relation: 抽取的关系（包含entity1_name, entity2_name, content）
            existing_relations: 已有关系列表（每个包含relation_id, content）
        
        Returns:
            如果匹配，返回 {"relation_id": "...", "need_update": True/False}
            如果不匹配，返回 None
        """
        if not existing_relations:
            return None
        
        system_prompt = """你是一个关系对齐系统。你的任务是判断新抽取的关系是否与已有关系相同或非常相似。
如果判断是相同或非常相似的关系，返回匹配的relation_id和是否需要更新。
如果判断是完全新的关系，返回null。"""
        
        existing_str = "\n".join([
            f"- relation_id: {r['relation_id']}, content: {r['content']}"
            for r in existing_relations
        ])
        
        # 兼容新旧格式：支持 entity1_name/entity2_name 和 entity1_name/entity2_name
        entity1_name = extracted_relation.get('entity1_name') or extracted_relation.get('entity1_name', '')
        entity2_name = extracted_relation.get('entity2_name') or extracted_relation.get('entity2_name', '')
        
        prompt = f"""新抽取的关系：
- entity1: {entity1_name}
- entity2: {entity2_name}
- content: {extracted_relation['content']}

---

已有关系列表：

{existing_str}

---

请判断新抽取的关系是否与已有关系中的某一个相同或非常相似。
如果匹配，返回JSON格式：{{"relation_id": "匹配的relation_id", "need_update": true/false}}
如果不匹配，返回：null

只输出JSON，不要其他文字："""
        
        response = self._call_llm(prompt, system_prompt)
        
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            result = json.loads(response)
            if result is None or result == "null":
                return None
            return result
        except json.JSONDecodeError:
            return None
    
    def merge_relation_content(self, old_content: str, new_content: str) -> str:
        """
        合并关系内容（更新时使用）
        
        Args:
            old_content: 旧的内容
            new_content: 新的内容
        
        Returns:
            合并后的内容
        """
        system_prompt = """你是一个内容总结系统。你的任务是基于旧版本和新版本的内容，重新总结生成一个完整、准确的关系描述。

重要：
- **必须只输出JSON格式，包含content字段，不要包含任何其他文字或markdown代码块**"""
        
        prompt = f"""旧版本内容：

{old_content}

---

新版本内容：

{new_content}

---

请基于上述两个版本的内容，重新总结生成一个完整、准确的关系描述。

重要要求：
{self.CONTENT_MERGE_REQUIREMENTS}

**重要：只输出JSON格式，格式如下：**
{{"content": "重新总结后的完整关系描述"}}

不要包含任何其他文字、说明或markdown代码块，只输出纯JSON。"""
        
        response = self._call_llm(prompt, system_prompt)
        
        # 尝试解析JSON响应
        try:
            # 尝试提取JSON部分（如果响应包含其他文本）
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            # 清理JSON字符串，修复常见错误
            response = self._clean_json_string(response)
            
            # 解析JSON
            result = json.loads(response)
            
            # 提取content字段
            if isinstance(result, dict) and 'content' in result:
                merged_content = str(result['content']).strip()
                # 如果content不为空，返回合并后的内容
                if merged_content:
                    return merged_content
            
            # 如果JSON格式不正确或content为空，回退到原始响应
            print(f"警告：关系内容合并返回的JSON格式不正确或content为空，使用原始响应")
            return response.strip()
            
        except json.JSONDecodeError as e:
            # JSON解析失败，尝试修复
            try:
                fixed_response = self._fix_json_errors(response)
                result = json.loads(fixed_response)
                if isinstance(result, dict) and 'content' in result:
                    merged_content = str(result['content']).strip()
                    if merged_content:
                        return merged_content
                print(f"警告：关系内容合并JSON解析失败（已尝试修复），使用原始响应: {e}")
                return response.strip()
            except:
                # 如果修复也失败，返回原始响应（去除可能的markdown代码块）
                print(f"警告：关系内容合并JSON解析失败，使用原始响应: {e}")
                cleaned_response = self._clean_markdown_code_blocks(response)
                return cleaned_response.strip()
    
    def merge_multiple_relation_contents(self, contents: List[str]) -> str:
        """
        合并多个关系内容（用于去重合并）
        
        Args:
            contents: 多个关系内容列表
        
        Returns:
            合并后的内容
        """
        if not contents:
            return ""
        
        if len(contents) == 1:
            return contents[0]
        
        system_prompt = """你是一个关系内容总结系统。你的任务是基于多个关系描述，重新总结生成一个完整、准确的关系描述。"""
        
        contents_str = "\n\n---\n\n".join([
            f"关系描述 {i+1}:\n{content}" 
            for i, content in enumerate(contents)
        ])
        
        prompt = f"""以下是同一实体对的多个关系描述：

{contents_str}

---

请基于上述多个关系描述，重新总结生成一个完整、准确的关系描述。

重要要求：
1. **不是简单合并**：不要只是把多个描述拼接在一起，而是要基于所有描述的信息，重新组织和总结
2. **综合信息**：整合所有描述中的有效信息，包括不同方面、不同角度的描述
3. **重新组织**：用新的语言重新组织这些信息，生成一个连贯、完整、准确的描述
4. **去除冗余**：去除重复和冗余的信息，确保描述简洁而完整
5. **保持准确性**：确保总结后的内容准确反映所有描述中的重要信息
6. **完整性**：确保所有独特的信息都被保留在总结中

输出重新总结后的完整内容："""
        
        return self._call_llm(prompt, system_prompt)
    
    def enhance_entity_content(self, memory_cache: MemoryCache, input_text: str,
                              entity: Dict[str, str]) -> str:
        """
        实体后验增强：结合缓存记忆和当前text对实体的content进行更细致的补全挖掘
        
        Args:
            memory_cache: 当前的记忆缓存
            input_text: 当前窗口的输入文本
            entity: 实体字典，包含name和content
        
        Returns:
            增强后的实体content
        """
        system_prompt = """你是一个实体内容增强系统。你的任务是结合记忆缓存和当前文本，对已抽取的实体内容进行更细致的补全和挖掘。

任务要求：
1. 仔细分析记忆缓存中的相关信息（当前摘要等）
2. 结合当前输入文本，挖掘更多关于该实体本身的细节信息
3. 对实体的content进行补全、细化和增强，形成更完整的实体概要性总结
4. 保留原有信息，同时添加新发现的关于实体本身的细节信息
5. 确保增强后的内容更加完整、准确和丰富

content增强重点：
- 重点补充和细化实体的属性、介绍、特点等实体本身的描述
- 增加的细节应该是对实体本身的描述，包括实体的类型、属性、特征、特点等
- 避免添加实体与其他实体的关系或交互信息，专注于实体自身的描述
- 形成一个整体的概要性总结，全面概括该实体的核心信息

重要：
- 不要改变实体的基本定义和核心信息
- 只添加和补全信息，不要删除原有信息
- 基于记忆缓存和当前文本中确实存在的信息进行增强
- 不要编造或推测不存在的信息
- **必须只输出JSON格式，包含content字段，不要包含任何其他文字或markdown代码块**"""
        
        prompt = f"""当前记忆缓存：

{memory_cache.content}

---

当前输入文本：

{input_text}

---

已抽取的实体：

实体名称：{entity['name']}
当前content：{entity['content']}

---

请结合记忆缓存和当前输入文本，对该实体的content进行更细致的补全和挖掘。

要求：
1. 仔细分析记忆缓存中与该实体相关的信息（当前摘要等）
2. 从当前输入文本中挖掘更多关于该实体本身的细节、属性、特点等信息
3. 对实体的content进行补全、细化和增强，形成更完整的实体概要性总结
4. 保留原有信息，同时添加新发现的关于实体本身的细节信息
5. 确保增强后的内容更加完整、准确和丰富

content增强重点：
- 重点补充和细化实体的属性、介绍、特点等实体本身的描述
- 增加的细节应该是对实体本身的描述，包括实体的类型、属性、特征、特点等
- 避免添加实体与其他实体的关系或交互信息，专注于实体自身的描述
- 形成一个整体的概要性总结，全面概括该实体的核心信息

注意：
- 不要改变实体的基本定义和核心信息
- 只添加和补全信息，不要删除原有信息
- 基于记忆缓存和当前文本中确实存在的信息进行增强
- 不要编造或推测不存在的信息

**重要：只输出JSON格式，格式如下：**
{{"content": "增强后的完整实体content"}}

不要包含任何其他文字、说明或markdown代码块，只输出纯JSON。"""
        
        response = self._call_llm(prompt, system_prompt)
        
        # 尝试解析JSON响应
        try:
            # 尝试提取JSON部分（如果响应包含其他文本）
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            # 清理JSON字符串，修复常见错误
            response = self._clean_json_string(response)
            
            # 解析JSON
            result = json.loads(response)
            
            # 提取content字段
            if isinstance(result, dict) and 'content' in result:
                enhanced_content = str(result['content']).strip()
                # 如果content不为空，返回增强后的内容
                if enhanced_content:
                    return enhanced_content
            
            # 如果JSON格式不正确或content为空，回退到原始响应
            print(f"警告：实体后验增强返回的JSON格式不正确或content为空，使用原始响应")
            return response.strip()
            
        except json.JSONDecodeError as e:
            # JSON解析失败，尝试修复
            try:
                fixed_response = self._fix_json_errors(response)
                result = json.loads(fixed_response)
                if isinstance(result, dict) and 'content' in result:
                    enhanced_content = str(result['content']).strip()
                    if enhanced_content:
                        return enhanced_content
                print(f"警告：实体后验增强JSON解析失败（已尝试修复），使用原始响应: {e}")
                return response.strip()
            except:
                # 如果修复也失败，返回原始响应（去除可能的markdown代码块）
                print(f"警告：实体后验增强JSON解析失败，使用原始响应: {e}")
                cleaned_response = self._clean_markdown_code_blocks(response)
                return cleaned_response.strip()
    
    # ========== 知识图谱整理相关方法 ==========
    
    def analyze_entity_candidates_preliminary(self, entities_group: List[Dict[str, Any]], 
                                              content_snippet_length: int = 64) -> Dict[str, Any]:
        """
        初步筛选：分析一组候选实体，返回可能需要合并或存在关系的候选列表
        
        这是两步判断流程的第一步，使用截断的content进行快速筛选。
        
        Args:
            entities_group: 候选实体组，每个实体包含:
                - entity_id: 实体ID
                - name: 实体名称
                - content: 实体内容描述
                - version_count: 该实体的版本数量
            content_snippet_length: 传入LLM的实体content最大长度（默认64字符）
        
        Returns:
            初步筛选结果，包含:
            - possible_merges: 可能需要合并的实体对列表
            - possible_relations: 可能存在关系的实体对列表
            - no_action: 不需要处理的实体ID列表
        """
        if not entities_group or len(entities_group) < 2:
            return {"possible_merges": [], "possible_relations": [], "no_action": []}
        
        system_prompt = f"""你是一个知识图谱整理系统。你的任务是对候选实体进行**初步筛选**。

**任务说明**：
- 输入的第一个实体是**当前分析的实体**
- 其他实体是通过相似度搜索找到的**候选实体**
- 这是初步筛选阶段，你只需要判断哪些候选实体**可能**需要进一步分析
- **不要做最终决策**，只需要筛选出可能有关系的候选

**筛选标准**：

1. **可能需要合并**（possible_merges）：
   - 名称相同或非常相似
   - content描述看起来可能是同一个对象
   - 需要进一步用完整信息确认

2. **可能存在关系**（possible_relations）：
   - 名称或content中有明显的关联线索
   - 可能是别名、称呼关系
   - 可能有其他明确的关联
   - 需要进一步确认关系类型

3. **不需要处理**（no_action）：
   - 明显是不同的对象
   - 没有任何关联的线索

输出JSON格式，不要包含任何其他文字：
{{
  "possible_merges": [
    {{
      "entity_id": "候选实体的entity_id",
      "reason": "简要说明为什么可能需要合并"
    }}
  ],
  "possible_relations": [
    {{
      "entity_id": "候选实体的entity_id",
      "reason": "简要说明为什么可能存在关系"
    }}
  ],
  "no_action": ["不需要处理的候选实体entity_id列表"],
  "analysis_summary": "初步筛选总结"
}}

注意：
- 这是初步筛选，宁可多选也不要漏选
- 如果不确定，放入possible_relations中等待进一步判断
- 只判断当前实体（实体1）与候选实体之间的关系"""
        
        # 构建实体信息字符串
        entities_str = ""
        current_entity = entities_group[0]
        
        entities_str += f"""
【当前分析的实体】实体1:
- entity_id: {current_entity.get('entity_id', '')}
- name: {current_entity.get('name', '')}
- version_count: {current_entity.get('version_count', 1)}
- content: {current_entity.get('content', '')[:content_snippet_length]}{'...' if len(current_entity.get('content', '')) > content_snippet_length else ''}
"""
        
        for i, entity in enumerate(entities_group[1:], 2):
            content = entity.get('content', '')
            content_snippet = content[:content_snippet_length] + ('...' if len(content) > content_snippet_length else '')
            
            entities_str += f"""
【候选实体】实体{i}:
- entity_id: {entity.get('entity_id', '')}
- name: {entity.get('name', '')}
- version_count: {entity.get('version_count', 1)}
- content: {content_snippet}
"""
        
        prompt = f"""请对以下实体进行初步筛选，判断哪些候选实体**可能**与当前实体有关联：

{entities_str}

请分析每个候选实体，将它们分类到possible_merges、possible_relations或no_action中。

只输出JSON，不要其他文字："""
        
        # 调用LLM
        try:
            response = self._call_llm(prompt, system_prompt)
            
            # 解析JSON响应
            json_str = response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            
            json_str = self._clean_json_string(json_str)
            result = json.loads(json_str)
            
            if not isinstance(result, dict):
                raise ValueError("响应格式不正确")
            
            # 确保必需的字段存在
            if "possible_merges" not in result:
                result["possible_merges"] = []
            if "possible_relations" not in result:
                result["possible_relations"] = []
            if "no_action" not in result:
                result["no_action"] = []
            
            return result
            
        except Exception as e:
            print(f"  初步筛选出错: {e}")
            # 出错时返回所有候选都可能有关系，以便后续精细化判断
            return {
                "possible_merges": [],
                "possible_relations": [{"entity_id": e.get("entity_id"), "reason": "初步筛选出错，需要精细化判断"} for e in entities_group[1:]],
                "no_action": [],
                "error": str(e)
            }
    
    def analyze_entity_pair_detailed(self, 
                                     current_entity: Dict[str, Any],
                                     candidate_entity: Dict[str, Any],
                                     existing_relations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        精细化判断：对一对实体进行详细分析，判断是否合并或创建关系
        
        这是两步判断流程的第二步，使用完整的content和已有关系进行精确判断。
        
        Args:
            current_entity: 当前实体，包含:
                - entity_id: 实体ID
                - name: 实体名称
                - content: 完整的实体内容描述
                - version_count: 版本数量
            candidate_entity: 候选实体，格式同上
            existing_relations: 两个实体之间已存在的关系列表，每个关系包含:
                - relation_id: 关系ID
                - content: 关系描述
        
        Returns:
            判断结果，包含:
            - action: "merge" | "create_relation" | "no_action"
            - reason: 判断理由
            - relation_content: 如果action是create_relation，提供关系描述
            - merge_target: 如果action是merge，提供目标entity_id
        """
        # 构建已有关系的提示
        existing_relations_note = ""
        if existing_relations and len(existing_relations) > 0:
            existing_relations_note = f"""
**⚠️ 这两个实体之间已存在以下关系**：
"""
            for rel in existing_relations:
                existing_relations_note += f"- {rel.get('content', '无描述')}\n"
            existing_relations_note += """
**重要**：已存在关系表明这两个实体之前被判断为不同实体。除非有明确证据证明它们确实是同一对象，否则不应该合并。
"""
        
        system_prompt = f"""你是一个知识图谱整理系统。你的任务是对两个实体进行**精细化判断**。

{self.ENTITY_ALIGNMENT_CORE_PRINCIPLES}

**判断标准**：

1. **合并（merge）**：两个实体描述的是**完全相同的对象**
   - 必须有明确证据证明是同一对象
   - 如果有任何疑问，不要合并
   
2. **创建关系（create_relation）**：两个实体是不同对象，但存在**明确的、有意义的关联**
   - 别名/称呼关系
   - 使用关系、从属关系、交互关系等
   - 关联必须是明确的、直接的
   
3. **不处理（no_action）**：两个实体无关，或关联不明确
   - 不同的对象且没有明确关联
   - 关联模糊或间接
{existing_relations_note}
输出JSON格式，不要包含任何其他文字：
{{
  "action": "merge" | "create_relation" | "no_action",
  "reason": "详细的判断理由",
  "relation_content": "如果action是create_relation，填写关系描述；否则留空",
  "merge_target": "如果action是merge，填写应该保留的entity_id（版本数多的）；否则留空"
}}"""
        
        prompt = f"""请对以下两个实体进行精细化判断：

**当前实体**：
- entity_id: {current_entity.get('entity_id', '')}
- name: {current_entity.get('name', '')}
- version_count: {current_entity.get('version_count', 1)}
- content: {current_entity.get('content', '')}

**候选实体**：
- entity_id: {candidate_entity.get('entity_id', '')}
- name: {candidate_entity.get('name', '')}
- version_count: {candidate_entity.get('version_count', 1)}
- content: {candidate_entity.get('content', '')}

请仔细分析这两个实体的**完整content描述**，判断：
1. 它们是否描述的是**完全相同的对象**？如果是，应该合并
2. 它们是否是不同对象但存在**明确的、有意义的关联**？如果是，应该创建关系
3. 它们是否无关或关联不明确？如果是，不处理

**判断时请注意**：
- content描述是最重要的判断依据
- 即使名称相似，如果content描述的是不同对象，不能合并
- 合并的标准是"完全相同的对象"，有任何疑问都不要合并
- 版本数多的实体应该作为合并的目标

只输出JSON，不要其他文字："""
        
        try:
            response = self._call_llm(prompt, system_prompt)
            
            # 解析JSON响应
            json_str = response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            
            json_str = self._clean_json_string(json_str)
            result = json.loads(json_str)
            
            if not isinstance(result, dict):
                raise ValueError("响应格式不正确")
            
            # 确保必需的字段存在
            if "action" not in result:
                result["action"] = "no_action"
            if "reason" not in result:
                result["reason"] = ""
            if "relation_content" not in result:
                result["relation_content"] = ""
            if "merge_target" not in result:
                result["merge_target"] = ""
            
            return result
            
        except Exception as e:
            print(f"  精细化判断出错: {e}")
            return {
                "action": "no_action",
                "reason": f"判断出错: {str(e)}",
                "relation_content": "",
                "merge_target": "",
                "error": str(e)
            }
    
    def analyze_entity_duplicates(self, entities_group: List[Dict[str, Any]], 
                                  memory_contexts: Optional[Dict[str, str]] = None,
                                  content_snippet_length: int = 64,
                                  existing_relations_between_entities: Optional[Dict[str, List[Dict]]] = None) -> Dict[str, Any]:
        """
        分析一组候选实体，判断是否为同一实体或存在别名关系（保留兼容性）
        
        注意：这是旧版本的方法，建议使用两步判断流程：
        1. analyze_entity_candidates_preliminary - 初步筛选
        2. analyze_entity_pair_detailed - 精细化判断
        
        Args:
            entities_group: 候选实体组，每个实体包含:
                - entity_id: 实体ID
                - name: 实体名称
                - content: 实体内容描述
                - version_count: 该实体的版本数量
            memory_contexts: 可选的记忆上下文字典，key是entity_id，value是对应的缓存记忆文本
            content_snippet_length: 传入LLM的实体content最大长度（默认64字符）
            existing_relations_between_entities: 已存在关系的实体对信息字典，
                key为 "entity1_id|entity2_id" 格式（按字母序排序），
                value为该实体对之间的关系列表，每个关系包含:
                    - relation_id: 关系ID
                    - content: 关系描述
        
        Returns:
            分析结果，包含:
            - merge_groups: 需要合并的实体组列表
            - alias_relations: 需要创建的别名关系列表
        """
        if not entities_group or len(entities_group) < 2:
            return {"merge_groups": [], "alias_relations": []}
        
        # 构建已有关系的提示信息
        existing_relations_note = ""
        if existing_relations_between_entities:
            existing_relations_note = """

**⚠️ 已存在关系的实体对（绝对不能合并！）**：
以下实体对之间**已经存在直接关联关系**，这表明它们是**不同的实体**，**绝对不能合并**！
对于这些实体对，你只需要判断：
- 如果有新的关系信息需要补充，放入 alias_relations 中（用于更新或新增关系）
- 如果没有新信息，则不需要处理

"""
            for pair_key, relations in existing_relations_between_entities.items():
                entity_ids = pair_key.split("|")
                if len(relations) > 0:
                    rel_contents = [r.get('content', '无描述')[:50] for r in relations[:3]]
                    existing_relations_note += f"- 实体 {entity_ids[0]} 和 {entity_ids[1]} 之间已有关系：{'; '.join(rel_contents)}\n"

        system_prompt = f"""你是一个知识图谱整理系统。你的任务是分析**当前实体**与**候选实体**之间的关系。

**任务说明**：
- 输入的第一个实体是**当前分析的实体**
- 其他实体是通过相似度搜索找到的**候选实体**
- 你需要判断：**当前实体与每个候选实体之间**是否需要合并或存在别名关系
- **不要判断候选实体之间的关系**，只关注当前实体与候选实体之间的关系

**关键原则：只有完全相同的对象才能合并！**
{existing_relations_note}
{self.ENTITY_ALIGNMENT_CORE_PRINCIPLES}

**扩展判断规则**：

1. **同一实体（可以合并）**：当前实体与候选实体描述的是**完全相同的对象**
   - ✅ 例如：当前实体"雷志成"和候选实体"雷志成"（重复创建，content描述的是同一个人）应该合并
   - ❌ **如果两个实体之间已存在关系，则它们是不同实体，绝对不能合并！**
   
2. **别名关系（创建关系边，可以考虑合并）**：当前实体与候选实体存在别名、称呼、简称、绰号等关系，且指向**同一个对象**
   - ✅ 例如：当前实体"雷政委"是候选实体"雷志成"的称呼（同一个人）应该创建别名关系，可以考虑合并
   - ❌ **如果两个实体之间已存在关系，说明它们已被识别为不同实体，不能合并！**

3. **确实有关联但不是同一对象（创建关系边，不合并）**：当前实体与候选实体**确实存在明确的、有意义的关联**，但它们是**不同的对象**
   - ✅ 关联必须是**明确的、直接的、有意义的**（例如：作者与作品、使用关系、从属关系等）
   - ❌ 不能是模糊的、间接的或牵强的关联（例如：只是在同一场景中出现但没有实际关联）

4. **无关实体（不合并，不创建关系）**：不同对象，且**没有明确的、有意义的关联**

**重要判断标准**：
- **必须仔细分析每个实体的content描述**，这是判断的关键依据
- **如果两个实体的content描述的是不同的对象**（即使名称相似或有关联），**绝对不能合并**
- **如果两个实体之间已存在关系，表明它们已被确认为不同实体，绝对不能合并！**
- 版本数量多的实体通常是"主实体"，应该作为合并的目标
- **有任何疑问时，宁可不合并，只创建关系边**
- **只判断当前实体（实体1）与候选实体（实体2-N）之间的关系**
- **如果不确定两个实体之间是否确实存在明确的、有意义的关联，宁可不创建关系边**

输出JSON格式，不要包含任何其他文字：
{{
  "merge_groups": [
    {{
      "target_entity_id": "保留的entity_id（版本数最多的）",
      "source_entity_ids": ["要合并过来的entity_id列表"],
      "reason": "合并理由（必须说明为什么是同一对象）"
    }}
  ],
  "alias_relations": [
    {{
      "entity1_id": "起始实体的entity_id",
      "entity2_id": "目标实体的entity_id", 
      "entity1_name": "起始实体名称",
      "entity2_name": "目标实体名称",
      "content": "初步的关系描述（简洁描述两个实体之间的关联，用于初步判断关系是否已存在）"
    }}
  ],
  "analysis_summary": "分析总结"
}}

注意：
- merge_groups：只有content明确描述是同一对象时才能合并，**且两个实体之间不存在已有关系**
- alias_relations：可以用于别名关系（同一对象的不同称呼），也可以用于普通关系（确实有关联但不同对象）
- alias_relations中的content字段是初步的关系描述，应该简洁但准确"""
        
        # 构建实体信息字符串
        # 第一个实体是当前分析的实体，其他是候选实体
        entities_str = ""
        for i, entity in enumerate(entities_group, 1):
            entity_id = entity.get('entity_id', '')
            name = entity.get('name', '')
            content = entity.get('content', '')
            version_count = entity.get('version_count', 1)
            
            # 标注当前实体和候选实体
            if i == 1:
                entity_label = "【当前分析的实体】"
            else:
                entity_label = "【候选实体】"
            
            # 限制content长度
            content_snippet = content[:content_snippet_length] + ('...' if len(content) > content_snippet_length else '')
            
            entities_str += f"""
{entity_label} 实体 {i}:
- entity_id: {entity_id}
- name: {name}
- version_count: {version_count}
- content: {content_snippet}
"""
        
        prompt = f"""请分析以下实体，判断**当前实体**（实体1）与**候选实体**（实体2-N）之间的关系：

{entities_str}

---

**重要说明**：
- **实体1**是当前正在分析的实体
- **实体2-N**是通过相似度搜索找到的候选实体
- 你需要判断：**实体1与每个候选实体之间**是否需要合并或存在别名关系
- **不要判断候选实体之间的关系**，只关注实体1与候选实体之间的关系

**⚠️ 关键判断标准**：
- **只有content描述的是完全相同的对象时，才能合并**
- **如果两个实体有关联但描述的是不同的对象，不应该合并，只应该创建关系边**
- **例如：《三体》（小说）和《科幻世界》（杂志）虽然有关联，但它们是不同的实体，不应该合并**

请仔细分析实体1与每个候选实体的名称和**content描述**，判断：
1. **实体1与哪些候选实体应该合并**（content明确描述的是同一对象）
2. **实体1与哪些候选实体存在别名关系**（一个是另一个的别名/称呼，且指向同一对象）
3. **实体1与哪些候选实体确实存在明确的、有意义的关联但不是同一对象**（**必须**创建关系边，但不合并）
4. **实体1与哪些候选实体是无关的**（不应该合并或创建关系）

**⚠️ 重要规则**：
- **仔细阅读每个实体的content描述**，这是判断的关键
- **如果content描述的是不同的对象，即使名称相似或有关联，也绝对不能合并**
- **如果两个实体之间已存在关系（见上方"已存在关系的实体对"），说明它们已被确认为不同实体，绝对不能合并！**
- **关键判断标准**：只有两个实体**确实存在明确的、直接的、有意义的关联**时，才应该在alias_relations中创建关系边
- **关联必须是明确的、直接的、有意义的**（例如：使用关系、包含关系、交互关系、从属关系等）
- **不能是模糊的、间接的或牵强的关联**（例如：只是因为在同一场景中出现，但没有实际的交互或联系）
- **如果不确定两个实体之间是否确实存在明确的、有意义的关联，宁可不创建关系边**
- 版本数量多的实体应该作为合并的目标（target）
- 如果实体1和候选实体是同一对象，应该合并（保留版本数多的作为target）
- **如果不确定是否为同一对象，宁可不合并，也不创建关系边**
- 合并和别名关系可以同时存在（先合并，再创建别名关系）

只输出JSON，不要其他文字："""
        
        # 重试机制：最多重试3次
        max_retries = 3
        last_error = None
        last_response = None
        
        for attempt in range(max_retries):
            try:
                # 调用LLM
                response = self._call_llm(prompt, system_prompt)
                last_response = response
                
                # 解析JSON响应
                # 尝试提取JSON部分
                json_str = response
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                elif "```" in response:
                    json_start = response.find("```") + 3
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                
                # 清理JSON字符串
                json_str = self._clean_json_string(json_str)
                
                # 尝试解析JSON
                result = json.loads(json_str)
                
                # 验证结果格式
                if not isinstance(result, dict):
                    raise ValueError("响应格式不正确：不是字典类型")
                
                # 确保必需的字段存在
                if "merge_groups" not in result:
                    result["merge_groups"] = []
                if "alias_relations" not in result:
                    result["alias_relations"] = []
                
                # 验证和清理alias_relations中的数据
                valid_alias_relations = []
                for rel in result.get("alias_relations", []):
                    if isinstance(rel, dict):
                        entity1_id = rel.get("entity1_id")
                        entity2_id = rel.get("entity2_id")
                        # 如果entity_id为空或None，跳过这个关系
                        if entity1_id and entity2_id:
                            valid_alias_relations.append(rel)
                        else:
                            # 记录警告但不跳过，尝试后续通过名称查找
                            print(f"    警告：alias_relation缺少entity_id: {rel}")
                            valid_alias_relations.append(rel)
                result["alias_relations"] = valid_alias_relations
                
                # 解析成功，返回结果
                if attempt > 0:
                    print(f"  重试成功（第 {attempt + 1} 次尝试）")
                return result
                
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_retries - 1:
                    # 尝试修复JSON错误
                    try:
                        fixed_response = self._fix_json_errors(json_str if 'json_str' in locals() else response)
                        result = json.loads(fixed_response)
                        if isinstance(result, dict):
                            if "merge_groups" not in result:
                                result["merge_groups"] = []
                            if "alias_relations" not in result:
                                result["alias_relations"] = []
                            print(f"  通过修复JSON错误成功（第 {attempt + 1} 次尝试）")
                            return result
                    except:
                        pass
                
                # 如果修复失败且还有重试机会，继续重试
                if attempt < max_retries - 1:
                    print(f"  解析实体分析JSON失败（第 {attempt + 1}/{max_retries} 次尝试）: {e}")
                    print(f"  响应内容: {response[:500] if 'response' in locals() else 'N/A'}...")
                    print(f"  正在重试...")
                    continue
                else:
                    # 最后一次尝试也失败
                    print(f"  解析实体分析JSON失败（已重试 {max_retries} 次）: {e}")
                    print(f"  响应内容: {last_response[:500] if last_response else 'N/A'}...")
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"  分析出错（第 {attempt + 1}/{max_retries} 次尝试）: {e}")
                    print(f"  正在重试...")
                    continue
                else:
                    print(f"  分析出错（已重试 {max_retries} 次）: {e}")
        
        # 所有重试都失败，返回错误
        error_msg = str(last_error) if last_error else "未知错误"
        return {"merge_groups": [], "alias_relations": [], "error": f"重试 {max_retries} 次后仍然失败: {error_msg}"}
    
    def generate_consolidation_summary(self, merge_results: List[Dict], 
                                       alias_results: List[Dict],
                                       analyzed_groups_count: int) -> str:
        """
        生成知识图谱整理的摘要
        
        Args:
            merge_results: 合并操作结果列表
            alias_results: 别名关系创建结果列表
            analyzed_groups_count: 分析的实体组数量
        
        Returns:
            整理摘要的Markdown格式文本
        """
        system_prompt = """你是一个知识图谱管理系统。请根据提供的整理操作结果，生成一个简洁的Markdown格式摘要。"""
        
        # 构建操作详情
        operations_str = f"分析的相似实体组数量: {analyzed_groups_count}\n\n"
        
        if merge_results:
            operations_str += "## 实体合并操作\n\n"
            for i, merge in enumerate(merge_results, 1):
                operations_str += f"{i}. 将 {merge.get('merged_source_ids', [])} 合并到 {merge.get('target_entity_id', '')}\n"
                operations_str += f"   - 更新的实体记录数: {merge.get('entities_updated', 0)}\n"
                if merge.get('reason'):
                    operations_str += f"   - 原因: {merge.get('reason', '')}\n"
        else:
            operations_str += "## 实体合并操作\n\n无需合并的实体。\n"
        
        if alias_results:
            operations_str += "\n## 别名关系创建\n\n"
            for i, alias in enumerate(alias_results, 1):
                operations_str += f"{i}. {alias.get('entity1_name', '')} -> {alias.get('entity2_name', '')}\n"
                operations_str += f"   - 关系描述: {alias.get('content', '')}\n"
        else:
            operations_str += "\n## 别名关系创建\n\n无需创建的别名关系。\n"
        
        prompt = f"""以下是知识图谱整理操作的详情：

{operations_str}

请生成一个简洁的Markdown格式摘要，包括：
1. 整理概述
2. 主要操作
3. 整理效果

直接输出Markdown内容，不要包含代码块标记："""
        
        response = self._call_llm(prompt, system_prompt)
        
        # 清理可能的代码块标记
        response = self._clean_markdown_code_blocks(response)
        
        return response
    
    def generate_relation_memory_cache(self, relations: List[Dict], 
                                       entities_info: List[Dict],
                                       entity_memory_caches: Dict[str, str]) -> str:
        """
        根据关系和实体生成记忆缓存内容
        
        Args:
            relations: 关系列表，每个包含 entity1_id, entity2_id, entity1_name, entity2_name, content, relation_id, is_new, is_updated
            entities_info: 实体信息列表，每个包含 entity_id, name, content
            entity_memory_caches: 实体ID到其记忆缓存内容的映射
        
        Returns:
            生成的记忆缓存内容（Markdown格式）
        """
        system_prompt = """你是一个知识图谱管理系统。你的任务是根据实体和关系信息，生成一份记忆缓存文档，记录系统当前正在处理的内容。

记忆缓存应该包含：
1. 当前活动：系统正在进行的操作（知识图谱整理、创建/更新关联关系等）
2. 处理内容：正在处理哪些实体和关系
3. 实体信息：涉及的主要实体及其描述
4. 关系信息：正在创建或更新的关联关系
5. 处理摘要：总结当前处理的内容和目的

重要：
- 用Markdown格式输出
- 保持简洁但信息完整
- 说明清楚系统在做什么，处理什么内容
- 不要包含实体ID或关系ID，只使用实体名称和关系描述"""
        
        # 构建实体信息
        entities_str = ""
        for entity_info in entities_info:
            entity_id = entity_info.get("entity_id", "")
            name = entity_info.get("name", "")
            content = entity_info.get("content", "")
            memory_cache_content = entity_memory_caches.get(entity_id, "")
            
            entities_str += f"\n### 实体: {name}\n"
            entities_str += f"- 实体描述: {content}\n"
            if memory_cache_content:
                entities_str += f"- 相关记忆缓存: {memory_cache_content[:200]}...\n"
        
        # 构建关系信息
        relations_str = ""
        for relation in relations:
            entity1_name = relation.get("entity1_name", "")
            entity2_name = relation.get("entity2_name", "")
            content = relation.get("content", "")
            is_new = relation.get("is_new", False)
            is_updated = relation.get("is_updated", False)
            
            status = "新建" if is_new else ("更新" if is_updated else "已存在")
            relations_str += f"\n- {entity1_name} -> {entity2_name} ({status})\n"
            relations_str += f"  关系描述: {content}\n"
        
        prompt = f"""系统正在处理以下实体和关系：

## 涉及的实体
{entities_str}

## 正在处理的关系
{relations_str}

请生成一份记忆缓存文档，说明：
1. 系统当前正在进行的操作
2. 正在处理哪些实体和关系
3. 这些实体和关系的内容和意义
4. 处理的目的和结果

直接输出Markdown内容，不要包含代码块标记："""
        
        response = self._call_llm(prompt, system_prompt)
        
        # 清理可能的代码块标记
        response = self._clean_markdown_code_blocks(response)
        
        return response.strip()
    
    def judge_need_create_relation(self, entity1_name: str, entity1_content: str,
                                    entity2_name: str, entity2_content: str,
                                    entity1_memory_cache: Optional[str] = None,
                                    entity2_memory_cache: Optional[str] = None) -> bool:
        """
        判断两个实体之间是否真的需要创建关系边
        
        这个方法在生成关系content之前调用，使用两个实体的完整信息（name、content、memory_cache）
        来判断是否确实存在明确的、有意义的关联。
        
        Args:
            entity1_name: 起始实体名称
            entity1_content: 起始实体内容描述
            entity2_name: 目标实体名称
            entity2_content: 目标实体内容描述
            entity1_memory_cache: 起始实体的记忆缓存内容（可选）
            entity2_memory_cache: 目标实体的记忆缓存内容（可选）
        
        Returns:
            True表示确实需要创建关系边，False表示不需要
        """
        system_prompt = f"""你是一个关系判断系统。你的任务是根据两个实体的完整信息，判断它们之间是否确实存在明确的、有意义的关联，需要创建关系边。

重要判断标准：
{self.RELATION_VALIDITY_CRITERIA}

**必须只输出JSON格式，包含need_create字段，不要包含任何其他文字或markdown代码块**"""
        
        # 构建实体信息
        entities_info = f"""
起始实体：
- 名称: {entity1_name}
- 描述: {entity1_content}
"""
        if entity1_memory_cache:
            entities_info += f"- 相关记忆缓存: {entity1_memory_cache[:500]}...\n"
        
        entities_info += f"""
目标实体：
- 名称: {entity2_name}
- 描述: {entity2_content}
"""
        if entity2_memory_cache:
            entities_info += f"- 相关记忆缓存: {entity2_memory_cache[:500]}...\n"
        
        prompt = f"""请根据以下两个实体的完整信息，判断它们之间是否确实存在明确的、有意义的关联，需要创建关系边：

{entities_info}

请仔细分析：
1. 两个实体的名称和描述
2. 记忆缓存中提到的相关信息
3. 它们之间是否存在明确的、直接的、有意义的关联

判断标准：
- ✅ **需要创建关系边**：如果两个实体之间存在明确的、直接的、有意义的关联（例如：使用关系、包含关系、交互关系、从属关系等）
- ❌ **不需要创建关系边**：如果两个实体之间只是模糊的、间接的或牵强的关联（例如：只是因为在同一场景中出现，但没有实际的交互或联系）

**重要：只输出JSON格式，格式如下：**
{{"need_create": true/false}}

不要包含任何其他文字、说明或markdown代码块，只输出纯JSON。"""
        
        response = self._call_llm(prompt, system_prompt)
        
        # 尝试解析JSON响应
        try:
            # 尝试提取JSON部分
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            # 清理JSON字符串
            response = self._clean_json_string(response)
            
            result = json.loads(response)
            
            if isinstance(result, dict) and "need_create" in result:
                return bool(result["need_create"])
            else:
                # 如果解析失败，默认返回False（保守策略）
                return False
                
        except json.JSONDecodeError as e:
            print(f"解析判断关系JSON失败: {e}")
            print(f"响应内容: {response[:500]}...")
            
            # 尝试修复
            try:
                fixed_response = self._fix_json_errors(response)
                result = json.loads(fixed_response)
                if isinstance(result, dict) and "need_create" in result:
                    return bool(result["need_create"])
                else:
                    return False
            except:
                # 如果修复失败，默认返回False（保守策略）
                return False
    
    def generate_relation_content(self, entity1_name: str, entity1_content: str,
                                  entity2_name: str, entity2_content: str,
                                  relation_memory_cache: Optional[str] = None,
                                  preliminary_content: Optional[str] = None) -> str:
        """
        根据两个实体和memory_cache生成关系的content
        
        Args:
            entity1_name: 起始实体名称
            entity1_content: 起始实体内容描述
            entity2_name: 目标实体名称
            entity2_content: 目标实体内容描述
            relation_memory_cache: 关系的记忆缓存内容（可选，这是总结两个实体的memory_cache）
            preliminary_content: 初步的关系content（可选，用于参考）
        
        Returns:
            生成的关系content（自然语言描述）
        """
        system_prompt = """你是一个知识图谱关系生成系统。你的任务是根据两个实体的信息和相关记忆缓存，生成一个准确、完整的关系描述。

重要：
- **必须只输出JSON格式，包含content字段，不要包含任何其他文字或markdown代码块**
- **关系描述必须专注于描述两个实体之间的关系，不要包含实体自身的其他信息**
- 如果提供了初步关系描述，应该基于它来生成更详细的关系描述
- 关系描述应该简洁但完整，能够清楚地说明两个实体之间的关系"""
        
        # 构建初步关系描述信息
        preliminary_info = ""
        if preliminary_content:
            preliminary_info = f"""
## 初步关系描述（作为参考）：
{preliminary_content}

**注意：请基于这个初步描述，生成更详细的关系描述，但必须专注于两个实体之间的关系，不要添加实体自身的其他不相关信息。**
"""
        
        # 构建实体信息（简化，只提供基本信息）
        entities_info = f"""
起始实体：
- 名称: {entity1_name}
- 描述: {entity1_content[:200]}{'...' if len(entity1_content) > 200 else ''}

目标实体：
- 名称: {entity2_name}
- 描述: {entity2_content[:200]}{'...' if len(entity2_content) > 200 else ''}
"""
        
        # 构建关系记忆缓存信息（只关注与关系相关的部分）
        relation_context = ""
        if relation_memory_cache:
            relation_context = f"""
## 关系记忆缓存（与两个实体关系相关的信息）：
{relation_memory_cache[:500]}{'...' if len(relation_memory_cache) > 500 else ''}
"""
        
        prompt = f"""请根据以下信息生成两个实体之间的关系描述：

{preliminary_info}
{entities_info}
{relation_context}

**重要要求：**
1. **专注于关系**：只描述两个实体之间的关系，不要包含实体自身的其他信息（如职业、经历等与关系无关的信息）
2. **基于初步描述**：如果提供了初步关系描述，请基于它来生成更详细的关系描述
3. **简洁准确**：关系描述应该简洁但准确，直接说明两个实体之间的关系

生成一个准确、专注的关系描述，只说明这两个实体之间的关系。

**重要：只输出JSON格式，格式如下：**
{{"content": "关系描述"}}

不要包含任何其他文字、说明或markdown代码块，只输出纯JSON。"""
        
        response = self._call_llm(prompt, system_prompt)
        
        # 尝试解析JSON响应
        try:
            # 尝试提取JSON部分
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            # 清理JSON字符串
            response = self._clean_json_string(response)
            
            result = json.loads(response)
            
            if isinstance(result, dict) and "content" in result:
                return result["content"]
            else:
                # 如果解析失败，返回默认描述
                return f"'{entity1_name}'与'{entity2_name}'的关联关系"
                
        except json.JSONDecodeError as e:
            print(f"解析关系content JSON失败: {e}")
            # 显示更多调试信息
            error_pos = getattr(e, 'pos', None)
            if error_pos is not None:
                start = max(0, error_pos - 100)
                end = min(len(response), error_pos + 100)
                print(f"错误位置: {error_pos}, 附近内容: {response[start:end]}")
            else:
                print(f"响应内容前500字符: {response[:500]}...")
            
            # 尝试修复
            try:
                fixed_response = self._fix_json_errors(response)
                result = json.loads(fixed_response)
                if isinstance(result, dict) and "content" in result:
                    print(f"修复成功，使用修复后的JSON")
                    return result["content"]
                else:
                    print(f"修复后JSON格式不正确，使用默认描述")
                    return f"'{entity1_name}'与'{entity2_name}'的关联关系"
            except Exception as fix_error:
                print(f"修复JSON失败: {fix_error}")
                # 如果修复也失败，尝试直接从响应中提取content字段的值（使用正则表达式）
                try:
                    import re
                    # 尝试用正则表达式提取content字段的值
                    content_match = re.search(r'"content"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', response)
                    if content_match:
                        content_value = content_match.group(1)
                        # 解码转义字符
                        content_value = content_value.encode().decode('unicode_escape')
                        print(f"使用正则表达式提取的content: {content_value[:100]}...")
                        return content_value
                except Exception as regex_error:
                    print(f"正则表达式提取失败: {regex_error}")
                
                # 最后回退到默认描述
                print(f"所有修复尝试都失败，使用默认描述")
                return f"'{entity1_name}'与'{entity2_name}'的关联关系"