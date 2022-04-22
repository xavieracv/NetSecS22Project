import pandas as pd
import regex as re

def remove_tld(domain):
  p = re.compile('.+(\.[^\.]+)')
  m = p.match(domain)
  if m is not None:
    tld = m.group(1)
    rm_num = len(tld)
    return domain[:-rm_num]
  else:
    return domain
  

def remove_tld_from_csv(csv_file, savename):
  """
  Args:
    csv_file: (str) the file to remove top level domains from
                    expects that there is a column "domain"
    savename: (str) the file name to save processed domain data to
  """
  data = pd.read_csv(csv_file, header=[0])
  data['domain'] = data['domain'].transform(remove_tld)
  data.to_csv(savename)

if __name__ == "__main__":
    path = "./"
    remove_tld_from_csv(
        path +"test_combined_multiclass.csv",
        path +"processed_test_combined_multiclass.csv"
    )