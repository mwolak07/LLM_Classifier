import torch.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import float16


prompt = "Using only the following context:\n" \
         "With a degree in the physical sciences, you can also become an engineer. PayScale.com reported that the 10th-90th percentiles salary range for entry-level civil engineers was $42,568-$69,295 in February 2014.ith your degree, you might become a fish and game warden or a fish and wildlife manager. Based on May 2012 salary figures from the U.S. Bureau of Labor Statistics (BLS), fish and game wardens in the 10th-90th percentile range earned yearly wages between $31,870 and $70,750 (www.bls.gov).\n" \
         "\n" \
         "If you're planning on pursuing a master's degree, you're probably wondering what type of salary you might earn. The average starting salary for a graduate with a master's degree varies greatly depending on the subject of study and chosen career. Read on to see what your field of interest pays entry-level employees.ith your degree, you might become a fish and game warden or a fish and wildlife manager. Based on May 2012 salary figures from the U.S. Bureau of Labor Statistics (BLS), fish and game wardens in the 10th-90th percentile range earned yearly wages between $31,870 and $70,750 (www.bls.gov).\n" \
         "\n" \
         "The average starting salary for individuals with a master's degree depends on their field of study, among other factors.Keep reading to learn about some of the average starting salaries for graduates of master's degree programs in such fields as business, engineering, computer science and the humanities.he average starting salary for individuals with a master's degree depends on their field of study, among other factors.\n" \
         "\n" \
         "Other positions that are directly related to a psychology degree can pay better. For example, the median salary for a social worker in 2010, according to the Bureau of Labor Statistics, was $42,480.Those working in human resources in 2010 earned median salaries of $52,690.ther positions that are directly related to a psychology degree can pay better. For example, the median salary for a social worker in 2010, according to the Bureau of Labor Statistics, was $42,480.\n" \
         "\n" \
         "A doctorate (from Latin docere, to teach) or doctor's degree (from Latin doctor, teacher) or doctoral degree is an academic degree (Ph.D. or Ed.D.) awarded by universities that, in most countries, qualifies the holder to teach at the university level in the degree's field, or to work in a specific profession. doctorate (from Latin docere, to teach) or doctor's degree (from Latin doctor, teacher) or doctoral degree is an academic degree (Ph.D. or Ed.D.) awarded by universities that, in most countries, qualifies the holder to teach at the university level in the degree's field, or to work in a specific profession.\n" \
         "\n" \
         "Career Options. Salaries of recent graduates with math degrees will vary based on their position. In addition, your location and employer all have an effect on earnings. Some entry-level jobs one can attain with a bachelor's degree in the field include actuary, mathematician, federal statistician and math teacher.hile many mathematician jobs require a master's degree or higher, entry-level positions with the federal government can be attained with a bachelor's degree.\n" \
         "\n" \
         "Doctorate Degree. The median income of a psychologist who was licensed and held a doctorate degree in 2010 was $68,640, according to the Bureau of Labor Statistics. The median salary for industrial-organizational psychologists was $87,330.The median salary for clinical, counseling and school psychologists was $66,810.ther positions that are directly related to a psychology degree can pay better. For example, the median salary for a social worker in 2010, according to the Bureau of Labor Statistics, was $42,480.\n" \
         "\n" \
         "PayScale.com reported in April 2014 that most actuaries with a Bachelor of Science in Mathematics earned between $49,741 and $146,188. While many mathematician jobs require a master's degree or higher, entry-level positions with the federal government can be attained with a bachelor's degree.hile many mathematician jobs require a master's degree or higher, entry-level positions with the federal government can be attained with a bachelor's degree.\n" \
         "\n" \
         "The average graduate starting salary is a fairly robust \u00a325,000, though we\u2019ve also seen folk reporting anything between \u00a316,000 and \u00a370,000! Who gets what depends on what subject you study, what industry you go into and even where in the country your dream job is based.Cue our guide to first-job wages. first salary in Retail Management will likely be in the range of \u00a312,000-\u00a322,000, but some graduate training schemes pay handsomely for impressive candidates. Budget supermaket chain Aldi is a go-to for its grad scheme, which pays \u00a342,000 in the first year.\n" \
         "\n" \
         "The average starting salary for master's degree graduates can vary greatly based off of several factors, including the type of degree earned. For example, a computer science program graduate typically earns more than someone who graduates from a sociology program, even though they both hold a master's degree.he average starting salary for individuals with a master's degree depends on their field of study, among other factors.\n" \
         "\n" \
         "The answer, in complete sentences, to the question, \"entry level income after doctorate degree?\", is:\n",


def test_model(model_name):
    """
     0.) 
     1.) 
     2.) 
     3.) 
     4.) 
     5.) 
     6.) 
     7.) 
     8.) 
     9.) 
    10.) 
    11.) 
    12.) 
    13.) 
    14.) 
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_input = tokenizer(prompt)
    encoded_output = model.generate(encoded_input.input_ids, max_length=250)
    output = tokenizer.batch_decode(encoded_output)
    print(output)


def test_model_gpu(model_name):
    """
     0.) 
     1.) 
     2.) 
     3.) 
     4.) 
     5.) 
     6.) 
     7.) 
     8.) 
     9.) 
    10.) 
    11.) 
    12.) 
    13.) 
    14.) 
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=float16)
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_input = tokenizer(prompt)
    encoded_output = model.generate(encoded_input.input_ids.to(torch.device('cuda')), max_length=250)
    output = tokenizer.batch_decode(encoded_output)
    print(output)


model_names = [
    'EleutherAI/gpt-neo-2.7B', 
    'gpt2-xl',
    'bigscience/bloom-3b', 
    'facebook/opt-2.7b', 
    'EleutherAI/gpt-neo-1.3B', 
    'gpt2-large', 
    'facebook/opt-1.3b',
    'bigscience/bloom-1b1', 
    'gpt2-medium', 
    'bigscience/bloom-560m', 
    'facebook/opt-350m', 
    'EleutherAI/gpt-neo-125m', 
    'gpt2', 
    'facebook/opt-125m'
]
test_model(model_names[0])
