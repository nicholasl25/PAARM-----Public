"""### Filter Class Definition (Helper Functions: filter_keys)"""

# Class that represents a filter of what keys should and should not be included in the report
# typ: list of strings that represents what types should be included in the report DOM/INT
# airports: list of strings that represents the names of all airports to be included in the report
# airlines: a boolean that is True if the report should be divided by airline and False if it should not

class Filter:
  # Class defintion
  def __init__(self, typ, airports, airlines):
      self.typ = typ
      self.airports = airports
      self.airlines = airlines

  # Returns True if key should be included in the report and False otherwise
  # checks key[0] is a valid type, key[1] is a valid airport, and key[2] is a valid airline where 
  # key is a list of Strings in ["type", "airport", "airline"] form

  def filter_keys(self, key):
     type_bool = ((key[0] in self.typ) or ('both' in self.typ))
     airport_bool = ((key[1] in self.airports) or ('all' in self.airports))
     airline_bool = self.airlines or key[2] == "Subtotal"
     return type_bool and airport_bool and airline_bool