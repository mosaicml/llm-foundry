# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest
import transformers

from llmfoundry import TiktokenTokenizerWrapper

TEST_STRINGS = [
    'Hello world!',
    'def hello_world(input: str):\n    print(input)',
    '0000000000000000000000000000',
    '19234324 asas sf 119aASDFM AW3RAW-AF;;9900',
    '\n\n\n\nhello\n\t,'
]

# taken from https://github.com/explosion/spaCy/blob/8f0d6b0a8c42e4852bf6e24cdf629043f2f39361/spacy/tests/tokenizer/test_naughty_strings.py#L7
NAUGHTY_STRINGS = [
    # ASCII punctuation
    r",./;'[]\-=",
    r'<>?:"{}|_+',
    r'!@#$%^&*()`~"',
    # Unicode additional control characters, byte order marks
    r"­؀؁؂؃؄؅؜۝܏᠎​‌‍‎‏‪",
    r"￾",
    # Unicode Symbols
    r"Ω≈ç√∫˜µ≤≥÷",
    r"åß∂ƒ©˙∆˚¬…æ",
    "œ∑´®†¥¨ˆøπ“‘",
    r"¡™£¢∞§¶•ªº–≠",
    r"¸˛Ç◊ı˜Â¯˘¿",
    r"ÅÍÎÏ˝ÓÔÒÚÆ☃",
    r"Œ„´‰ˇÁ¨ˆØ∏”’",
    r"`⁄€‹›ﬁﬂ‡°·‚—±",
    r"⅛⅜⅝⅞",
    r"ЁЂЃЄЅІЇЈЉЊЋЌЍЎЏАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя",
    r"٠١٢٣٤٥٦٧٨٩",
    # Unicode Subscript/Superscript/Accents
    r"⁰⁴⁵",
    r"₀₁₂",
    r"⁰⁴⁵₀₁₂",
    r"ด้้้้้็็็็็้้้้้็็็็็้้้้้้้้็็็็็้้้้้็็็็็้้้้้้้้็็็็็้้้้้็็็็็้้้้้้้้็็็็็้้้้้็็็็ ด้้้้้็็็็็้้้้้็็็็็้้้้้้้้็็็็็้้้้้็็็็็้้้้้้้้็็็็็้้้้้็็็็็้้้้้้้้็็็็็้้้้้็็็็ ด้้้้้็็็็็้้้้้็็็็็้้้้้้้้็็็็็้้้้้็็็็็้้้้้้้้็็็็็้้้้้็็็็็้้้้้้้้็็็็็้้้้้็็็็",
    r" ̄  ̄",
    # Two-Byte Characters
    r"田中さんにあげて下さい",
    r"パーティーへ行かないか",
    r"和製漢語",
    r"部落格",
    r"사회과학원 어학연구소",
    r"찦차를 타고 온 펲시맨과 쑛다리 똠방각하",
    r"社會科學院語學研究所",
    r"울란바토르",
    r"𠜎𠜱𠝹𠱓𠱸𠲖𠳏",
    # Japanese Emoticons
    r"ヽ༼ຈل͜ຈ༽ﾉ ヽ༼ຈل͜ຈ༽ﾉ",
    r"(｡◕ ∀ ◕｡)",
    r"｀ｨ(´∀｀∩",
    r"__ﾛ(,_,*)",
    r"・(￣∀￣)・:*:",
    r"ﾟ･✿ヾ╲(｡◕‿◕｡)╱✿･ﾟ",
    r",。・:*:・゜’( ☻ ω ☻ )。・:*:・゜’",
    r"(╯°□°）╯︵ ┻━┻)" "(ﾉಥ益ಥ）ﾉ﻿ ┻━┻", # type: ignore
    r"┬─┬ノ( º _ ºノ)",
    r"( ͡° ͜ʖ ͡°)",
    # Emoji
    r"😍",
    r"👩🏽",
    r"👾 🙇 💁 🙅 🙆 🙋 🙎 🙍",
    r"🐵 🙈 🙉 🙊",
    r"❤️ 💔 💌 💕 💞 💓 💗 💖 💘 💝 💟 💜 💛 💚 💙",
    r"✋🏿 💪🏿 👐🏿 🙌🏿 👏🏿 🙏🏿",
    r"🚾 🆒 🆓 🆕 🆖 🆗 🆙 🏧",
    r"0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ 🔟",
    # Regional Indicator Symbols
    r"🇺🇸🇷🇺🇸 🇦🇫🇦🇲🇸",
    r"🇺🇸🇷🇺🇸🇦🇫🇦🇲",
    r"🇺🇸🇷🇺🇸🇦",
    # Unicode Numbers
    r"１２３",
    r"١٢٣",
    # Right-To-Left Strings
    r"ثم نفس سقطت وبالتحديد،, جزيرتي باستخدام أن دنو. إذ هنا؟ الستار وتنصيب كان. أهّل ايطاليا، بريطانيا-فرنسا قد أخذ. سليمان، إتفاقية بين ما, يذكر الحدود أي بعد, معاملة بولندا، الإطلاق عل إيو.",
    r"إيو.",
    r"בְּרֵאשִׁית, בָּרָא אֱלֹהִים, אֵת הַשָּׁמַיִם, וְאֵת הָאָרֶץ",
    r"הָיְתָהtestالصفحات التّحول",
    r"﷽",
    r"ﷺ",
    r"مُنَاقَشَةُ سُبُلِ اِسْتِخْدَامِ اللُّغَةِ فِي النُّظُمِ الْقَائِمَةِ وَفِيم يَخُصَّ التَّطْبِيقَاتُ الْحاسُوبِيَّةُ،",
    # Trick Unicode
    r"‪‪test‪",
    r"‫test",
    r" test ",
    r"test⁠test",
    r"⁦test⁧",
    # Zalgo Text
    r"Ṱ̺̺̕o͞ ̷i̲̬͇̪͙n̝̗͕v̟̜̘̦͟o̶̙̰̠kè͚̮̺̪̹̱̤ ̖t̝͕̳̣̻̪͞h̼͓̲̦̳̘̲e͇̣̰̦̬͎ ̢̼̻̱̘h͚͎͙̜̣̲ͅi̦̲̣̰̤v̻͍e̺̭̳̪̰-m̢iͅn̖̺̞̲̯̰d̵̼̟͙̩̼̘̳ ̞̥̱̳̭r̛̗̘e͙p͠r̼̞̻̭̗e̺̠̣͟s̘͇̳͍̝͉e͉̥̯̞̲͚̬͜ǹ̬͎͎̟̖͇̤t͍̬̤͓̼̭͘ͅi̪̱n͠g̴͉ ͏͉ͅc̬̟h͡a̫̻̯͘o̫̟̖͍̙̝͉s̗̦̲.̨̹͈̣",
    r"̡͓̞ͅI̗̘̦͝n͇͇͙v̮̫ok̲̫̙͈i̖͙̭̹̠̞n̡̻̮̣̺g̲͈͙̭͙̬͎ ̰t͔̦h̞̲e̢̤ ͍̬̲͖f̴̘͕̣è͖ẹ̥̩l͖͔͚i͓͚̦͠n͖͍̗͓̳̮g͍ ̨o͚̪͡f̘̣̬ ̖̘͖̟͙̮c҉͔̫͖͓͇͖ͅh̵̤̣͚͔á̗̼͕ͅo̼̣̥s̱͈̺̖̦̻͢.̛̖̞̠̫̰",
    r"̗̺͖̹̯͓Ṯ̤͍̥͇͈h̲́e͏͓̼̗̙̼̣͔ ͇̜̱̠͓͍ͅN͕͠e̗̱z̘̝̜̺͙p̤̺̹͍̯͚e̠̻̠͜r̨̤͍̺̖͔̖̖d̠̟̭̬̝͟i̦͖̩͓͔̤a̠̗̬͉̙n͚͜ ̻̞̰͚ͅh̵͉i̳̞v̢͇ḙ͎͟-҉̭̩̼͔m̤̭̫i͕͇̝̦n̗͙ḍ̟ ̯̲͕͞ǫ̟̯̰̲͙̻̝f ̪̰̰̗̖̭̘͘c̦͍̲̞͍̩̙ḥ͚a̮͎̟̙͜ơ̩̹͎s̤.̝̝ ҉Z̡̖̜͖̰̣͉̜a͖̰͙̬͡l̲̫̳͍̩g̡̟̼̱͚̞̬ͅo̗͜.̟",
    r"̦H̬̤̗̤͝e͜ ̜̥̝̻͍̟́w̕h̖̯͓o̝͙̖͎̱̮ ҉̺̙̞̟͈W̷̼̭a̺̪͍į͈͕̭͙̯̜t̶̼̮s̘͙͖̕ ̠̫̠B̻͍͙͉̳ͅe̵h̵̬͇̫͙i̹͓̳̳̮͎̫̕n͟d̴̪̜̖ ̰͉̩͇͙̲͞ͅT͖̼͓̪͢h͏͓̮̻e̬̝̟ͅ ̤̹̝W͙̞̝͔͇͝ͅa͏͓͔̹̼̣l̴͔̰̤̟͔ḽ̫.͕",
    r"Z̮̞̠͙͔ͅḀ̗̞͈̻̗Ḷ͙͎̯̹̞͓G̻O̭̗̮",
    # Unicode Upsidedown
    r"˙ɐnbᴉlɐ ɐuƃɐɯ ǝɹolop ʇǝ ǝɹoqɐl ʇn ʇunpᴉpᴉɔuᴉ ɹodɯǝʇ poɯsnᴉǝ op pǝs 'ʇᴉlǝ ƃuᴉɔsᴉdᴉpɐ ɹnʇǝʇɔǝsuoɔ 'ʇǝɯɐ ʇᴉs ɹolop ɯnsdᴉ ɯǝɹo˥",
    r"00˙Ɩ$-",
    # Unicode font
    r"Ｔｈｅ ｑｕｉｃｋ ｂｒｏｗｎ ｆｏｘ ｊｕｍｐｓ ｏｖｅｒ ｔｈｅ ｌａｚｙ ｄｏｇ",
    r"𝐓𝐡𝐞 𝐪𝐮𝐢𝐜𝐤 𝐛𝐫𝐨𝐰𝐧 𝐟𝐨𝐱 𝐣𝐮𝐦𝐩𝐬 𝐨𝐯𝐞𝐫 𝐭𝐡𝐞 𝐥𝐚𝐳𝐲 𝐝𝐨𝐠",
    r"𝕿𝖍𝖊 𝖖𝖚𝖎𝖈𝖐 𝖇𝖗𝖔𝖜𝖓 𝖋𝖔𝖝 𝖏𝖚𝖒𝖕𝖘 𝖔𝖛𝖊𝖗 𝖙𝖍𝖊 𝖑𝖆𝖟𝖞 𝖉𝖔𝖌",
    r"𝑻𝒉𝒆 𝒒𝒖𝒊𝒄𝒌 𝒃𝒓𝒐𝒘𝒏 𝒇𝒐𝒙 𝒋𝒖𝒎𝒑𝒔 𝒐𝒗𝒆𝒓 𝒕𝒉𝒆 𝒍𝒂𝒛𝒚 𝒅𝒐𝒈",
    r"𝓣𝓱𝓮 𝓺𝓾𝓲𝓬𝓴 𝓫𝓻𝓸𝔀𝓷 𝓯𝓸𝔁 𝓳𝓾𝓶𝓹𝓼 𝓸𝓿𝓮𝓻 𝓽𝓱𝓮 𝓵𝓪𝔃𝔂 𝓭𝓸𝓰",
    r"𝕋𝕙𝕖 𝕢𝕦𝕚𝕔𝕜 𝕓𝕣𝕠𝕨𝕟 𝕗𝕠𝕩 𝕛𝕦𝕞𝕡𝕤 𝕠𝕧𝕖𝕣 𝕥𝕙𝕖 𝕝𝕒𝕫𝕪 𝕕𝕠𝕘",
    r"𝚃𝚑𝚎 𝚚𝚞𝚒𝚌𝚔 𝚋𝚛𝚘𝚠𝚗 𝚏𝚘𝚡 𝚓𝚞𝚖𝚙𝚜 𝚘𝚟𝚎𝚛 𝚝𝚑𝚎 𝚕𝚊𝚣𝚢 𝚍𝚘𝚐",
    r"⒯⒣⒠ ⒬⒰⒤⒞⒦ ⒝⒭⒪⒲⒩ ⒡⒪⒳ ⒥⒰⒨⒫⒮ ⒪⒱⒠⒭ ⒯⒣⒠ ⒧⒜⒵⒴ ⒟⒪⒢",
    # File paths
    r"../../../../../../../../../../../etc/passwd%00",
    r"../../../../../../../../../../../etc/hosts",
    # iOS Vulnerabilities
    r"Powerلُلُصّبُلُلصّبُررً ॣ ॣh ॣ ॣ冗",
    r"🏳0🌈️",
]

TEST_STRINGS += NAUGHTY_STRINGS

@pytest.mark.parametrize('model_name',
                         ['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003'])
def test_tiktoken(model_name: str, tmp_path: pathlib.Path):
    tiktoken = pytest.importorskip('tiktoken')

    # Construction
    wrapped_tokenizer = TiktokenTokenizerWrapper(model_name='gpt-4')
    original_tokenizer = tiktoken.encoding_for_model('gpt-4')

    # Repr works
    _ = wrapped_tokenizer.__repr__()

    # Save and load
    wrapped_tokenizer.save_pretrained(tmp_path)
    reloaded_wrapped_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tmp_path, trust_remote_code=True)

    didnt_match = []
    # Simple tokenization test
    for string in TEST_STRINGS:
        wrapped_output = wrapped_tokenizer(string)
        original_output = original_tokenizer.encode(string)
        reloaded_wrapped_output = reloaded_wrapped_tokenizer(string)
        assert wrapped_output['input_ids'] == original_output
        assert set(wrapped_output.keys()) == {'input_ids', 'attention_mask'}
        assert reloaded_wrapped_output == wrapped_output

    # Round trip
    for string in TEST_STRINGS:
        wrapped_output = wrapped_tokenizer.decode(wrapped_tokenizer(string)['input_ids'])
        original_output = original_tokenizer.decode(original_tokenizer.encode(string))
        reloaded_wrapped_output = reloaded_wrapped_tokenizer.decode(reloaded_wrapped_tokenizer(string)['input_ids'])
        assert wrapped_output == string
        assert original_output == string
        assert reloaded_wrapped_output == string

    # Batched tokenization
    wrapped_output = wrapped_tokenizer(
        ['Hello world!', 'Hello world but longer!'])
    original_output = original_tokenizer.encode_batch(
        ['Hello world!', 'Hello world but longer!'])
    reloaded_wrapped_output = reloaded_wrapped_tokenizer(
        ['Hello world!', 'Hello world but longer!'])
    assert wrapped_output['input_ids'] == original_output
    assert set(wrapped_output.keys()) == {'input_ids', 'attention_mask'}
    assert reloaded_wrapped_output == wrapped_output

    # With padding
    wrapped_tokenizer.pad_token_id = wrapped_tokenizer.eos_token_id
    reloaded_wrapped_tokenizer.pad_token_id = reloaded_wrapped_tokenizer.eos_token_id
    wrapped_output = wrapped_tokenizer(
        ['Hello world!', 'Hello world but longer!'], padding=True)
    original_output = original_tokenizer.encode_batch(
        ['Hello world!', 'Hello world but longer!'])
    reloaded_wrapped_output = reloaded_wrapped_tokenizer(
        ['Hello world!', 'Hello world but longer!'], padding=True)
    for wrapped1, attn_mask, original1 in zip(wrapped_output['input_ids'],
                                              wrapped_output['attention_mask'],
                                              original_output):
        original_length = len(original1)
        assert wrapped1[:original_length] == original1
        assert sum(attn_mask) == original_length

    assert set(wrapped_output.keys()) == {'input_ids', 'attention_mask'}
    assert reloaded_wrapped_output == wrapped_output

    # Get vocab
    wrapped_vocab = wrapped_tokenizer.get_vocab()
    reloaded_wrapped_vocab = reloaded_wrapped_tokenizer.get_vocab()
    assert wrapped_vocab == reloaded_wrapped_vocab

    didnt_match = []
    for key, value in wrapped_vocab.items():
        if original_tokenizer.encode(key, allowed_special='all') == [value]:
            continue
        else:
            didnt_match.append(
                (key, original_tokenizer.encode(key,
                                                allowed_special='all'), value))

    # Decode is lossy because some bytes are not representable in utf-8
    # see https://github.com/openai/tiktoken/blob/39f29cecdb6fc38d9a3434e5dd15e4de58cf3c80/tiktoken/core.py#L245-L247
    assert len(didnt_match) == 77
